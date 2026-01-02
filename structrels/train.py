import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from utils import save_model
from evaluation import eval_model

from utils import obtain_w_and_b
from plot import plot_faithfulness_historgram

from src.operators import LinearRelationOperator


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_average_faithfulness(plot_dict, wandb_it):
    totals = {}
    counts = {}

    for metrics in plot_dict.values():
        split = metrics["split"]
        if split not in totals:
            totals[split] = {"tensor": 0, "jacobi": 0}
            counts[split] = 0
        totals[split]["tensor"] += metrics["tensor_faithfulness"]
        totals[split]["jacobi"] += metrics["jacobi_faithfulness"]
        counts[split] += 1

    log_payload = {}
    for split, split_totals in totals.items():
        count = counts[split]
        if count == 0:
            continue
        log_payload[f"{split}/tensor_{split}_mean"] = split_totals["tensor"] / count
        log_payload[f"{split}/jacobi_{split}_mean"] = split_totals["jacobi"] / count

    if log_payload:
        wandb.log(log_payload, wandb_it)


def get_loss(output_matrices, relations, mt, args, z_layer=-1):

    loss_fn = nn.CrossEntropyLoss()
    loss = torch.zeros(1,).to(mt.model.device)

    for output_matrix, relation in zip(output_matrices, relations):
        t_W, t_bias = obtain_w_and_b(output_matrix, mt.model.device)
        l2_reg = args.matrix_weight_decay * torch.sum(t_W ** 2)
        loss = loss + l2_reg

        h_layer = relation.properties.h_layer
        beta = relation.properties.beta
        operator = LinearRelationOperator(mt=mt, h_layer=h_layer, beta=beta, weight=t_W, z_layer=z_layer, bias=t_bias, prompt_template=relation.prompt_templates[0], use_cache=args.use_emb_cache)

        for sample in relation.samples:
            pred_token_dist = operator(subject=sample.subject).dist.to(mt.model.device)

            target_token = mt.tokenizer.encode(sample.object)[0]
            target_token_tensor = torch.tensor([target_token], dtype=torch.long).to(mt.model.device)

            loss += loss_fn(pred_token_dist, target_token_tensor)

    return loss


def training(model, train_dataset, test_dataset, num_iters, batch_size, optim_type, lr, device, args, mt = None, soft_labels=None):
    wandb_it = 0

    if optim_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.optim_weight_decay)
    elif optim_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.optim_weight_decay)
    elif optim_type =="SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.optim_weight_decay)
    else:
        raise NotImplementedError

    lr_sched = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=args.warmup_iters)

    num_params = count_parameters(model)
    print(f"Total number of parameters: {num_params}")
    wandb.log({"num_params": num_params})

    plot_loss_data = {}
    plot_loss_data['train'] = []
    plot_loss_data['test'] = []

    for epoch in range(num_iters):
        optimizer.zero_grad()

        # Generate a random batch from the dataset
        batch_indices = torch.randint(0, train_dataset.xs.shape[0], (batch_size,))

        input_vectors = train_dataset.xs[batch_indices].to(device)  # Shape: (batch_size, outer_dim)
        # target_matrices = train_dataset.ys[batch_indices].to(device)  # Shape: (batch_size, outer_dim, outer_dim)
        relations = []
        for i in batch_indices:
            relations.append(train_dataset.relations[i])

        output_matrices = model(input_vectors)  # Shape: (batch_size, outer_dim, outer_dim)

        train_loss = get_loss(output_matrices, relations, mt, args=args)

        train_loss.backward()
        wandb_it += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)

        optimizer.step()
        lr_sched.step()

        if (epoch + 1) % args.log_freq == 0:
            if args.do_test_eval == True:
                plot_dict = eval_model(args=args, checkpoint_epoch=epoch+1, split_by=args.split_by, model=model, embedding_dataset={"train": train_dataset, "test": test_dataset}, mt=mt, sample_cut=args.sample_cut_for_eval, wandb_it=wandb_it)
            else:
                plot_dict = eval_model(args=args, checkpoint_epoch=epoch+1, split_by=args.split_by, model=model, embedding_dataset={"train": train_dataset}, mt=mt, sample_cut=args.sample_cut_for_eval, wandb_it=wandb_it)

            plot_faithfulness_historgram(
                plot_dict=plot_dict,
                exp_name=args.exp_name,
                epoch=epoch + 1,
                wandb_it=wandb_it,
                results_dir=args.results_dir,
            )
            log_average_faithfulness(plot_dict, wandb_it)
            save_model(model=model, exp_name=args.exp_name, run_id=args.run_id, optimizer=optimizer, args=args, checkpoint_epoch=epoch+1)

    return plot_loss_data, args.log_freq, optimizer

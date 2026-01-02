import torch
import wandb

from collections import defaultdict
from typing import TypedDict

from src import data, functional

from src.operators import LinearRelationOperator
from utils import obtain_w_and_b


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


device = "cuda" if torch.cuda.is_available() else "cpu"


FaithfulnessDataDict = TypedDict('FaithfulnessDataDict', {'subject': str, 'object': str, 'pred': str, 'prob': float, 'tick_flag':str})

def check_faithfulness_for_model(expName: str, runId, device, args, embedding_dataset, model, mt, split_by, sample_cut=None, wandb_it=None):

    if split_by=="all_train" and not args.split_even_if_all_train:
        splits = ["train"]
    else:
        splits = ["train", "test"]

    plot_dict = {}

    for split in splits:

        for x,y,relation in embedding_dataset[split]:

            relation_embedding = torch.reshape(x, (1, model.outer_dim_r)).to(device)

            o_relation = y.to(device)
            o_W, o_bias = obtain_w_and_b(o_relation, device)

            t_relation = model(relation_embedding)[0].to(device)
            t_W, t_bias = obtain_w_and_b(t_relation, device)

            tensor_based_operator = LinearRelationOperator(mt=mt, h_layer=relation.properties.h_layer, beta=relation.properties.beta, weight=t_W, z_layer=relation.properties.z_layer, bias=t_bias, prompt_template=relation.prompt_templates[0])
            jacobi_based_operator = LinearRelationOperator(mt=mt, h_layer=relation.properties.h_layer, beta=relation.properties.beta, weight=o_W, z_layer=relation.properties.z_layer, bias=o_bias, prompt_template=relation.prompt_templates[0])

            tensor_faithfulness, tensor_data_list, tensor_pred_probs = check_faithfulness_for_relation_and_operator(relation=relation, operator=tensor_based_operator, sample_cut=sample_cut)
            jacobi_faithfulness, jacobi_data_list, jacobi_pred_probs = check_faithfulness_for_relation_and_operator(relation=relation, operator=jacobi_based_operator, sample_cut=sample_cut)

            most_common_object_faithfulness = get_most_common_object_faithfulness(relation=relation, mt=mt)

            wandb.log({f"{split}/{relation.name}/tensor faithfulness": tensor_faithfulness}, step=wandb_it)
            wandb.log({f"{split}/{relation.name}/jacobi faithfullness": jacobi_faithfulness}, step=wandb_it)

            wandb.log({f"{split}/{relation.name}/most common object faithfulness": most_common_object_faithfulness}, step=wandb_it)
            wandb.log({f"{split}/{relation.name}/tensor predictions": wandb.Histogram(tensor_pred_probs)}, step=wandb_it)
            wandb.log({f"{split}/{relation.name}/jacobi predictions": wandb.Histogram(jacobi_pred_probs)}, step=wandb_it)

            plot_dict[f"{relation.name}_{split}"] = {
                "split": split,
                "tensor_faithfulness": tensor_faithfulness,
                "jacobi_faithfulness": jacobi_faithfulness,
                "most_common_object_faithfulness": most_common_object_faithfulness
            }

    return plot_dict


def get_most_common_object_faithfulness(relation, mt):
    object_first_token_count = defaultdict(int)
    for sample in relation.samples:
        inputs = mt.tokenizer(sample.object, return_tensors="pt", padding="longest").to(
            mt.model.device
        )
        first_object_token = inputs["input_ids"][0][0].item()
        first_object_str = mt.tokenizer.decode(first_object_token)

        object_first_token_count[first_object_str] += 1

    most_common_object = max(object_first_token_count, key=object_first_token_count.get)
    most_common_object_count = object_first_token_count[most_common_object]

    acc = most_common_object_count / len(relation.samples)

    return acc


def check_faithfulness_for_relation_and_operator(relation: data.Relation, operator: LinearRelationOperator, sample_cut: int | None=None):

    data_list = []
    correct = 0
    wrong = 0

    if sample_cut is not None:
        sample_list = relation.samples[:sample_cut]
    else:
        sample_list = relation.samples

    pred_probs = []

    for sample in sample_list:
        predictions = operator(subject = sample.subject).predictions
        known_flag = functional.is_nontrivial_prefix(
            prediction=predictions[0].token, target=sample.object
        )

        correct += known_flag
        wrong += not known_flag

        DataDict = FaithfulnessDataDict({
            'subject': sample.subject,
            'object': sample.object,
            'pred': f"{functional.format_whitespace(predictions[0].token)}",
            'prob': predictions[0].prob,
            'tick_flag':f"{functional.get_tick_marker(known_flag)}"
        })
        data_list.append(DataDict)
        pred_probs.append(predictions[0].prob)

    faithfulness = correct/(correct + wrong)

    return faithfulness, data_list, pred_probs


def eval_model(args, checkpoint_epoch, split_by, model=None, embedding_dataset=None, mt=None, sample_cut=None, wandb_it=None):
    plot_dict = check_faithfulness_for_model(expName=args.exp_name, runId=args.run_id, device=device, model=model, args=args, embedding_dataset=embedding_dataset, mt=mt, split_by=split_by, sample_cut=sample_cut, wandb_it=wandb_it)
    return plot_dict

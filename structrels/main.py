import torch
import argparse
import os
import wandb

import data
import train
import utils

import src.models as models
from soft_labels import get_soft_labels
import tensor_networks

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    device = "cuda"

    if args.mt == "gptj":
        mt = models.load_model("gptj", device=device, fp16=False)
        r_dim, o_dim, s_dim = 4096, 4096, 4096
    elif args.mt == "neo" :
        mt = models.load_model("EleutherAI/gpt-neox-20b", device=device, fp16=False)
        r_dim, o_dim, s_dim = 6144, 6144, 6144
    elif args.mt == "llama31" :
        mt = models.load_model("meta-llama/Llama-3.1-8B", device=device, fp16=False)
        r_dim, o_dim, s_dim = 4096, 4096, 4096
    else:
        raise NotImplementedError


    # load the data
    logging.info("Load the data")
    splits = ["train", "test"]

    dataset_dict = data.make_dataset_splits(
        dataset_name="relation_embeddings",
        folder_path=args.folder_path,
        splits=splits,
        split_by=args.split_by,
        split_even_if_all_train=args.per_relation_split,
        relation_names_file=args.relation_names_file,
        mt=mt,
        relation_embeds=args.relation_embeds,
        r_dim=r_dim,
        per_relation_split_frac=args.per_relation_split_frac,
        split_seed=args.split_seed,
        split_by_relations_test_ratio=args.split_by_relations_test_ratio,
        hparams_path=args.hparams_path,
        decoders_path=args.decoders_path
    )
    train_dataset, test_dataset = dataset_dict["train"], dataset_dict["test"]

    with torch.no_grad():
        if args.soft_labels:
            soft_labels = get_soft_labels([train_dataset, test_dataset], mt, device)

    logging.info("Initialize tensor model")

    if args.model == "triangle":
        model = tensor_networks.TriangleNetworkRelationToMatrix(
            outer_dim=args.outer_dim+1,
            inn_s=args.inner_dim_s,
            inn_r=args.inner_dim_r,
            inn_o=args.inner_dim_o,
            inn_x=args.inn_x,
            inn_y=args.inn_y,
            inn_z=args.inn_z,
            with_nn=args.with_nn).to(device)
    elif args.model == "block":
        model = tensor_networks.SimpleBlockNetworkRelationToMatrix(
            outer_dim_s=s_dim+1,
            outer_dim_r=r_dim,
            outer_dim_o=o_dim+1,
            inner_dim_s=args.inner_dim_s,
            inner_dim_r=args.inner_dim_r,
            inner_dim_o=args.inner_dim_o,
            with_nn=args.with_nn).to(device)
    elif args.model == "individual_matrices":
        num_rels = len(train_dataset)
        model = tensor_networks.IndividualMatricesBlockNetworkRelationToMatrix(
            outer_dim_s=args.outer_dim+1,
            outer_dim_r=num_rels,
            outer_dim_o=args.outer_dim+1,
            with_nn=args.with_nn,
            rank=args.rank).to(device)

    # train the network
    logging.info("Starting training")
    train.training(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        lr=args.lr,
        optim_type=args.optim_type,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        device=device,
        args=args,
        mt=mt,
        soft_labels=soft_labels if args.soft_labels else None)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="relations_parser")
    parser.add_argument("--seed", type=int, default=None, help="seed")

    # network params
    parser.add_argument("--outer_dim", type=int, default=5, help="outer dimension")
    parser.add_argument("--with_nn", type=utils.str2bool, default=False, help="nn with relation leg")

    parser.add_argument("--inner_dim_s", type=int, default=20, help="inner dimension for subject leg")
    parser.add_argument("--inner_dim_r", type=int, default=20, help="inner dimension for relation leg")
    parser.add_argument("--inner_dim_o", type=int, default=20, help="inner dimension for object leg")
    parser.add_argument("--inn_x", type=int, default=20, help="inner dimension for x")
    parser.add_argument("--inn_y", type=int, default=20, help="inner dimension for y")
    parser.add_argument("--inn_z", type=int, default=20, help="inner dimension for z")

    # train params
    parser.add_argument("--split_by", type=str, default='relations', help="relations: split by relations, all_train: all relations are in the training set")
    parser.add_argument("--model", type=str, default='block', help="model type")
    parser.add_argument("--lr", type=float, default=0.1, help="learining rate")
    parser.add_argument("--matrix_weight_decay", type=float, default=0.00001, help="weight decay matrix")
    parser.add_argument("--optim_weight_decay", type=float, default=0.00001, help="weight decay for optim")
    parser.add_argument("--optim_type", type=str, default="SGD", help="optimizer")
    parser.add_argument("--num_iters", type=int, default=9000, help="number of iterations")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--num_train_relations_in_batch", type=int, default=10, help="number of train relations in a batch")
    parser.add_argument("--relation_names_file", type=str, help="relation names file")
    parser.add_argument("--folder_path", type=str, help="where the data .jsons are")
    parser.add_argument("--mt", type=str, default="gptj", help="mt type, gptj, neo or llama31")
    parser.add_argument("--relation_embeds", type=str, default=None, help="specifies how relation embeds are calculated")
    parser.add_argument("--so_embeds", type=str, default=None, help="specifies how so embeds are calculated")
    parser.add_argument("--rank", type=int, default=None, help="rank of the core tensor parametrization")
    parser.add_argument("--per_relation_split", type=utils.str2bool, default=False, help="enable per-relation train/test split even when split_by=all_train")
    parser.add_argument("--per_relation_split_frac", type=float, default=0.7, help="fraction of samples to keep in train when per_relation_split is enabled")
    parser.add_argument("--split_seed", type=int, default=None, help="seed for per-relation splitting")
    parser.add_argument("--warmup_iters", type=int, default=200, help="number of warmup iterations")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="max norm for gradient clipping")
    parser.add_argument("--sample_cut_for_eval", type=int, default=30, help="number of samples to eval faithfulness on, if None uses all samples")
    parser.add_argument("--soft_labels", type=utils.str2bool, default=False, help="whether to use soft labels during training")
    parser.add_argument("--decoders_path", type=str, default="decoders/", help="path to the decoders")
    parser.add_argument("--split_by_relations_test_ratio", type=float, default=0.25, help="when splitting by relations, the test ratio to use")
    parser.add_argument("--hparams_path", type=str, default="hparams/", help="path to the hparams")
    parser.add_argument("--use_emb_cache", type=utils.str2bool, default=True, help="use simple caching mechanism to compute embeddings")
    parser.add_argument("--split_even_if_all_train", type=utils.str2bool, default=False, help="enable sample-wise splitting even if all_train is selected")


    #logging params
    parser.add_argument("--tag", type=str, default="debug", help="tag for wandb run")
    parser.add_argument("--exp_name", type=str, default="exp1", help="name of the experiment")
    parser.add_argument("--run_id", type=str, default="0", help="run_id")
    parser.add_argument("--log_freq", type=int, default=1000, help="logging frequency")
    parser.add_argument("--do_test_eval", type=utils.str2bool, default=True, help="whether to do test eval during training")
    parser.add_argument("--results_dir", type=str, default="results", help="directory for output artifacts")

    args = parser.parse_args()

    if args.seed is not None:
        utils.set_seed(args.seed)

    wandb_project = os.environ.get("WANDB_PROJECT", "")
    wandb_entity = os.environ.get("WANDB_ENTITY", "")
    wandb.init(project=wandb_project, entity=wandb_entity, config=args, tags=[args.tag])

    main(args)

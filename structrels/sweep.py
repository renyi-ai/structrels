"""Run sweeps over different hyperparameters by relation."""
import argparse
import logging

import os
import sys

from src import data, hparams, models, sweeps
from src.utils import experiment_utils, logging_utils

import torch

logger = logging.getLogger(__name__)


def load_dataset_from_args(args: argparse.Namespace) -> data.RelationDataset:
    """Load a dataset based on args from `add_data_args`."""
    logger.info(args.load_data_from)
    dataset = data.load_dataset(args.load_data_from)
    dataset = dataset.filter(
        relation_names=args.rel_names,
        relation_type=args.rel_types,
        domain_name=args.rel_domains,
        range_name=args.rel_ranges,
        disambiguating=args.rel_disamb,
        symmetric=args.rel_sym,
        fn_type=args.rel_fn_types,
    )
    if len(dataset.relations) == 0:
        raise ValueError("no relations found matching all criteria")
    return dataset


def main(args: argparse.Namespace) -> None:
    logging_utils.configure(args)
    experiment = experiment_utils.setup_experiment(args)

    device = args.device or "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_dataset_from_args(args)
    # mt = models.load_model("gptj", device=device, fp16=False)

    if args.model == "gptj":
        mt = models.load_model("gptj", device=device, fp16=False)
    elif args.model == "neo" :
        mt = models.load_model("EleutherAI/gpt-neox-20b", device=device)
    elif args.model == "llama" :
        mt = models.load_model("meta-llama/Llama-2-13b-hf", device=device)
    elif args.model == "llama31" :
        mt = models.load_model("meta-llama/Llama-3.1-8B", device=device)
    else:
        raise NotImplementedError


    with torch.device(device):
        results = sweeps.sweep(
            mt=mt,
            dataset=dataset,
            h_layers=args.h_layers,
            n_trials=args.n_trials,
            n_train_samples=args.n_train_samples,
            recall_k=args.recall_k,
            batch_size=args.batch_size,
            results_dir=experiment.results_dir,
            resume=args.resume,
            subj_token_filter=args.subj_token_filter,
            use_bare_prompt=args.use_bare_prompt,
        )
        for relation in results.relations:
            log_msg = f"{relation.relation_name}"
            if len(relation.trials) < args.n_trials:
                log_msg += f" -- not enough number of trials ({len(relation.trials)} < {args.n_trials}) --> skipping"
                logger.info(log_msg)
                continue
            log_msg += f" (n_trials={len(relation.trials)})"
            logger.info(log_msg)
            best_by_f = relation.best_by_faithfulness()
            best_by_e = relation.best_by_efficacy()
            hparams.RelationHParams(
                relation_name=relation.relation_name,
                h_layer=best_by_f.layer,  # type: ignore
                h_layer_edit=best_by_e.layer,  # type: ignore
                z_layer=-1,
                beta=best_by_f.beta.mean,
                model_name=mt.name,
            ).save(
                os.path.join(args.hparams_path, mt.name, relation.relation_name + ".json")
            )

    results_file = experiment.results_dir / "results_all.json"
    results_file.parent.mkdir(exist_ok=True, parents=True)
    with results_file.open("w") as handle:
        handle.write(results.to_json(indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sweep over hyperparameters")
    data.add_data_args(parser)
    experiment_utils.add_experiment_args(parser)
    logging_utils.add_logging_args(parser)
    models.add_model_args(parser)
    parser.add_argument(
        "--h-layers", type=int, nargs="+", help="h layers to try, defaults to all"
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=sweeps.DEFAULT_RECALL_K,
        help="compute up to recall@k",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=sweeps.DEFAULT_BATCH_SIZE,
        help="max batch size for lm",
    )
    parser.add_argument(
        "--subj-token-filter",
        type=str,
        default="all",
        choices=["all", "multi", "single"],
        help="allows filtering out samples with multiple or single subj tokens. defaults to all",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=sweeps.DEFAULT_N_TRIALS,
        help="number of trials per relation",
    )
    parser.add_argument(
        "--n-train-samples",
        type=int,
        default=sweeps.DEFAULT_N_TRAIN_SAMPLES,
        help="number of train samples to use per trial",
    )
    parser.add_argument(
        "--limit-test-samples",
        type=int,
        default=None,
        help="number of test samples to use",
    )
    parser.add_argument(
        "--use-bare-prompt",
        action="store_true",
        default=False,
        help='will use bare prompt "{subj} {obj}"',
    )
    parser.add_argument(
        "--load-data-from",
        type=str,
        default="data",
        help='from where we load our data',
    )
    parser.add_argument(
        "--hparams-path",
        type=str,
        default="hparams",
        help='where we save hparams',
    )
    

    args = parser.parse_args()
    logger.info(args)
    main(args)

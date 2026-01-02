import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from collections import defaultdict
from typing import Iterable, Mapping, Sequence, TypedDict, Optional
import argparse
import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import utils

from src import data, functional, models  # noqa: E402
from src.operators import LinearRelationOperator  # noqa: E402


class FaithfulnessDataDict(TypedDict):
    subject: str
    object: str
    pred: str
    prob: float
    tick_flag: str


CrossFaithfulness = dict[str, dict[str, float]]
MatrixMap = dict[str, torch.Tensor]


def obtain_w_and_b(relation: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    w = relation[:, :-1].to(device)
    b = relation[:, -1].to(device)
    return w, b


def check_faithfulness_for_relation_and_operator(
    relation: data.Relation,
    operator: LinearRelationOperator,
    split: str,
    num_samples: int | None = None,
) -> tuple[float, list[FaithfulnessDataDict], list[float]]:
    """Checks the percentage of known relation samples predicted by `operator`."""
    data_list: list[FaithfulnessDataDict] = []
    correct = 0
    wrong = 0

    sample_list = relation.samples[:num_samples] if num_samples is not None else relation.samples
    pred_probs: list[float] = []

    if len(sample_list) == 0:
        return 0, [], []

    for sample in sample_list:
        predictions = operator(subject=sample.subject).predictions
        known_flag = functional.is_nontrivial_prefix(
            prediction=predictions[0].token, target=sample.object
        )
        correct += known_flag
        wrong += not known_flag

        data_dict = FaithfulnessDataDict(
            {
                "subject": sample.subject,
                "object": sample.object,
                "pred": f"{functional.format_whitespace(predictions[0].token)}",
                "prob": predictions[0].prob,
                "tick_flag": f"{functional.get_tick_marker(known_flag)}",
            }
        )
        data_list.append(data_dict)
        pred_probs.append(predictions[0].prob)

    logger.info("%s correct=%s wrong=%s", relation.name, correct, wrong)

    if correct + wrong == 0:
        return 0, [], []

    faithfulness = correct / (correct + wrong)
    return faithfulness, data_list, pred_probs


def load_all_matrices(decoder_folder: str, relations: Sequence[data.Relation], device: torch.device) -> MatrixMap:
    matrices: MatrixMap = {}
    for relation in relations:
        matrix_filename = relation.name.replace(" ", "_") + f"_layer={relation.properties.h_layer}.npy"
        matrix_filepath = os.path.join(decoder_folder, matrix_filename)

        if not os.path.exists(matrix_filepath):
            logger.info("Matrix file %s does not exist, skipping", matrix_filepath)
            continue

        logger.info("Loading matrix file %s", matrix_filepath)
        matrix = np.load(matrix_filepath)
        matrices[relation.name] = torch.as_tensor(matrix, device=device)

    return matrices


def make_heatmap_data(results: Mapping[str, Mapping[str, float]], relation_names: Sequence[str]) -> np.ndarray:
    """
    Build a dense matrix from nested dict results[const_rel][moving_rel] = faithfulness.
    """
    n = len(relation_names)
    heatmap_data = np.zeros((n, n), dtype=float)

    for i, const_rel in enumerate(relation_names):
        for j, moving_rel in enumerate(relation_names):
            heatmap_data[i, j] = results.get(const_rel, {}).get(moving_rel, np.nan)

    return heatmap_data


def plot_heatmap(heatmap_data: np.ndarray, relations: Sequence[str], moving_relations: Sequence[str], result_prefix: str) -> None:
    if len(relations) > 30:
        size_multiplier = 3
    elif len(relations) > 20:
        size_multiplier = 2
    else:
        size_multiplier = 1

    fig, ax = plt.subplots(figsize=(27 * size_multiplier, 21 * size_multiplier))

    sns.heatmap(
        heatmap_data,
        cmap="plasma",
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        xticklabels=relations,
        yticklabels=moving_relations,
        cbar_kws={
            "shrink": 1.0,
            "pad": 0.02,
        },
        ax=ax,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=18)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=18)
    ax.set_xlabel("Evaluated on Relation", fontsize=18 * size_multiplier * 3)
    ax.set_ylabel("Linear Decoder Approximated on Relation", fontsize=18 * size_multiplier * 3)

    plt.tight_layout()

    plt.savefig(f"{result_prefix}_cross_faithfulness.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{result_prefix}_cross_faithfulness.pdf", bbox_inches="tight")


def calculate_cross_faithfulness(
    matrices: MatrixMap,
    relations: Iterable[data.Relation],
    mt,
    device: torch.device,
    num_samples: int | None,
    args: argparse.Namespace,
) -> CrossFaithfulness:
    results: CrossFaithfulness = defaultdict(dict)

    logger.info("Calculating cross faithfulness")
    for const_rel in relations:
        for moving_rel in relations:
            logger.info("Calculating cross faithfulness for %s vs %s", const_rel.name, moving_rel.name)
            if const_rel.name not in matrices:
                logger.info("Skipping %s because matrix is missing", const_rel.name)
                continue

            o_relation = matrices[const_rel.name].to(device)
            o_W, o_bias = obtain_w_and_b(o_relation, device)

            if args.fp16:
                o_W = o_W.half()
                o_bias = o_bias.half()

            jacobi_based_operator = LinearRelationOperator(
                mt=mt,
                h_layer=const_rel.properties.h_layer,
                beta=const_rel.properties.beta,
                weight=o_W,
                z_layer=-1,
                bias=o_bias,
                prompt_template=moving_rel.prompt_templates[0],
            )
            jacobi_faithfulness, _, _ = check_faithfulness_for_relation_and_operator(
                relation=moving_rel,
                operator=jacobi_based_operator,
                split="train",
                num_samples=num_samples,
            )

            results[const_rel.name][moving_rel.name] = jacobi_faithfulness
    return results


def _device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-evaluate relation decoders.")
    parser.add_argument("--decoder_folder", type=str, default="../tensornetworks/matrices", help="Folder containing saved decoder matrices.")
    parser.add_argument("--shortlist_path", type=str, default=None, help="Optional shortlist file with relation names (one per line).")
    parser.add_argument("--hparams_path", type=str, default="../hparams", help="Optional path to hyperparameters file.")
    parser.add_argument("--data_path", type=str, default="../data", help="Path to the dataset folder.")
    parser.add_argument("--result_prefix", type=str, default="", help="Prefix for output files.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for output files.")
    parser.add_argument("--model", type=str, required=True, help="LLM to test on: gptj | neo | llama | llama31.")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of samples per relation when computing faithfulness.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto|cpu|cuda|cuda:0, etc.).")
    parser.add_argument("--fp16", type=utils.str2bool, default=False, help="whether to use fp16 precision")

    return parser.parse_args()


def load_model_by_name(model_name: str, device: torch.device, args: argparse.Namespace):
    if model_name == "gptj":
        return models.load_model("gptj", device=device, fp16=args.fp16)
    if model_name == "neo":
        return models.load_model("EleutherAI/gpt-neox-20b", device=device, fp16=args.fp16)
    if model_name == "llama31":
        return models.load_model("meta-llama/Llama-3.1-8B", device=device, fp16=args.fp16)
    raise NotImplementedError(f"Unsupported model: {model_name}")


def main() -> None:
    args = parse_args()
    device = _device_from_arg(args.device)

    mt = load_model_by_name(args.model, device=device, args=args)

    logger.info("Loading data from %s", args.data_path)
    dataset = data.load_dataset(args.data_path, hparams_path=args.hparams_path, model_short_name=mt.name)

    if args.shortlist_path is not None:
        with open(args.shortlist_path, "r", encoding="utf-8") as f:
            shortlist = f.read().splitlines()
            dataset = dataset.filter(relation_names=shortlist)
            logger.info("Shortlist applied: %d relations", len(dataset.relations))

    matrices = load_all_matrices(args.decoder_folder, dataset.relations, device=device)

    results = calculate_cross_faithfulness(
        matrices,
        dataset.relations,
        mt,
        device=device,
        num_samples=args.num_samples,
        args=args
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.result_prefix:
        result_prefix = results_dir / args.result_prefix
    else:
        result_prefix = results_dir / "cross_eval"

    result_prefix.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{result_prefix}_cross_eval_results.pkl", "wb") as f:
        pickle.dump(results, f)

    relation_names = [r.name for r in dataset.relations]
    heatmap_data = make_heatmap_data(results, relation_names)

    plot_heatmap(heatmap_data, relation_names, relation_names, result_prefix=str(result_prefix))


if __name__ == "__main__":
    main()

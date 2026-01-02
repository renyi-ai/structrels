# The Structure of Relation Decoding Linear Operators in Large Language Models

Supplementary material for the NeurIPS 2025 Spotlight paper [The Structure of Relation Decoding Linear Operators in Large Language Models](https://arxiv.org/abs/2510.26543).

This repository contains datasets and code for cross-evaluating relation decoders, and training tensor networks for relation decoding in large language models (LLMs). For full details, see the paper.


## Dataset

Datasets are provided in the data directory. See Appendix D of the paper for detailed descriptions.

- `data/extended`: Extended Dataset
- `data/math`: Mathematical Dataset
- `data/orig`: Original dataset from Hernandez et al.


## Code
### Requirements

- Python 3.10+
- `uv` for environment management
- CUDA-capable GPU recommended for sweeps and training

### Setup

1. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Sync the environment:

```bash
uv sync
```

### Data layout

- `data/`: relation JSON files and relation name lists (see `data/orig`, `data/extended`, `data/math`)
- `hparams/`: per-relation per-llm hyperparameters written by sweeps
- `results/`: sweep outputs, decoder matrices, and cross-eval results

### Usage

All example commands below are available as scripts in the `scripts` directory.

#### Sweep (find relation-specific hparams for different llms); also saves decoder matrices for cross-evaluation

```bash
bash scripts/sweep_example.sh
```

#### Cross-evaluation of learned decoders

```bash
bash scripts/cross_eval_example.sh
```

#### Tensor network training

Compression experiment (all relations in training set):

```bash
bash scripts/train_example.sh
```

Generalization experiment (held-out relations):

```bash
bash scripts/train_gen_example.sh
```


### Notes

- If you do not want to log to Weights & Biases, set `WANDB_MODE=disabled`.
- Copy `.env.sample` to `.env` and fill in W&B values as needed.

## Citation

```
@inproceedings{christ2025structure,
  title     = {The Structure of Relation Decoding Linear Operators in Large Language Models},
  author    = {Christ, Miranda Anna and Csisz{\'a}rik, Adri{\'a}n and Becs{\'o}, Gergely and Varga, D{\'a}niel},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  note      = {Spotlight paper}
}
```

## Licenses

This project is licensed under the MIT License.

It includes third-party code in `packages/relations`, which is also licensed
under the MIT License. See `THIRD_PARTY_NOTICES` for details.

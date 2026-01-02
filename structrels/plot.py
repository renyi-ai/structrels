import wandb
import os
import pickle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def extract_run_results(dataset, folder_path, split):
    run_results = {}

    for folder_name in os.listdir(folder_path):
        if dataset in folder_name:
            subfolder_path = os.path.join(folder_path, folder_name)
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                with open(file_path, "rb") as f:
                    run_data = pickle.load(f)

                key = (run_data["args"].inner_dim, run_data["args"].outer_dim)
                if key not in run_results or run_results[key][f"best_{split}_loss"] > run_data[f"best_{split}_loss"]:
                    run_results[key] = {
                        f"best_{split}_loss": run_data[f"best_{split}_loss"],
                        "lr": run_data["args"].lr,
                        "optim": run_data["args"].optim_type,
                        "theta": run_data["args"].theta,
                    }

    return run_results


def plot_faithfulness_historgram(plot_dict, exp_name, epoch, wandb_it, results_dir="results"):
    sorted_items = sorted(
        plot_dict.items(),
        key=lambda item: item[1]["tensor_faithfulness"],
        reverse=True
    )

    relation_names = [relation for relation, data in sorted_items]
    splits = [data["split"] for relation, data in sorted_items]
    tensor_values = [data["tensor_faithfulness"] for relation, data in sorted_items]
    jacobi_values = [data["jacobi_faithfulness"] for relation, data in sorted_items]
    most_common_values = [data["most_common_object_faithfulness"] for relation, data in sorted_items]

    colors = []
    edgecolors = []
    for split in splits:
        if split == "train":
            colors.append("cornflowerblue")
        else:
            colors.append("mediumpurple")

    fig, ax = plt.subplots(figsize=(21, 9))
    bars = ax.bar(relation_names, tensor_values, color=colors)
    plt.margins(x=0)

    for i, bar in enumerate(bars):
        x_center = bar.get_x() + bar.get_width() / 2
        y_jacobi = jacobi_values[i]
        y_most_common = most_common_values[i]
        x_start = x_center - bar.get_width() / 2
        x_end = x_center + bar.get_width() / 2

        ax.plot([x_start+0.1, x_end-0.1], [y_jacobi, y_jacobi], color='darkorange', linewidth=3)
        ax.plot([x_start+0.1, x_end-0.1], [y_most_common, y_most_common], color='limegreen', linewidth=3)

    ax.set_ylabel("Faithfulness", fontsize=18)
    ax.set_xlabel("Relations", fontsize=18)

    ax.set_ylim([0, 1])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    for i, bar in enumerate(bars):
        if jacobi_values[i] > tensor_values[i]:
            x_center = bar.get_x() + bar.get_width() / 2
            y_top = jacobi_values[i]
            ax.plot([x_center, x_center], [0, y_top], color='lavender', linestyle='-', linewidth=1, zorder=0)


    ax.grid(axis='y', color='lavender', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)

    train_patch = mpatches.Patch(color='cornflowerblue', label='Train relations')
    test_patch = mpatches.Patch(color='mediumpurple', label='Test relations')
    jacobi_line = Line2D([0], [0], color='darkorange', lw=2, label='Linear decoder faithfulness')
    most_common_line = Line2D([0], [0], color='limegreen', lw=2, label='Majority guess')

    ax.legend(handles=[train_patch, test_patch, jacobi_line, most_common_line], loc='upper right', handlelength=2, fontsize=15)

    plt.tight_layout()

    if wandb.run:
        wandb.log({f"Faithfulness @ epoch {epoch+1}": wandb.Image(fig)}, step=wandb_it)

    save_dir = os.path.join(results_dir, "faithfulness", f"{exp_name}_epoch_{epoch+1}")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "faithfulness_fig.png"))
    plt.savefig(os.path.join(save_dir, "faithfulness_fig.pdf"))

    plt.close(fig)


if __name__ == "__main__":
    import random

    splits = ["train", "test"]
    plot_dict = {}
    for i in range(0, 30):
        relation_name = f"relation_{i}"
        plot_dict[relation_name] = {
            "split": "test",
            "tensor_faithfulness": round(random.random(), 2),
            "jacobi_faithfulness": round(random.random(), 2),
            "most_common_object_faithfulness": round(random.random(), 2),
        }

    plot_faithfulness_historgram(plot_dict, "plot", 0, 0)

import torch
import numpy as np
import random
import os


def set_seed(seed):
    torch.manual_seed(seed)          # Set seed for PyTorch
    np.random.seed(seed)             # Set seed for NumPy
    random.seed(seed)

    if torch.cuda.is_available():    # If using CUDA
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')


def save_model(model, exp_name, run_id, optimizer, args, checkpoint_epoch):
    exp_name = "./results/models/" + exp_name + f"_epoch_{checkpoint_epoch}"
    os.makedirs(exp_name, exist_ok=True)
    file_path = os.path.join(exp_name, f"{run_id}.pt")

    torch.save({
        "model": model,
        "state_dict": model.state_dict(),
        "optim": optimizer,
        "epoch": checkpoint_epoch,
        "args": args,
    }, file_path)


def obtain_w_and_b(relation, device):
    # tensor wieghts
    w = relation[:relation.size(0)-1,:relation.size(1)-1]
    # tensor bias
    b = relation[:relation.size(0)-1,-1].to(device)

    return w, b

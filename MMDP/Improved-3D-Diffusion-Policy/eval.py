"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""
import os
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import pathlib
from omegaconf import OmegaConf
import hydra
from termcolor import cprint

import numpy as np
import matplotlib.pyplot as plt
from diffusion_policy_3d.common.pytorch_util import dict_apply
import torch

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['WANDB_SILENT'] = "True"

def plot_result(model_outputs, ground_truths, idx):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(model_outputs[:, 0], label='Model Output - Speed', color='blue', marker='o', linestyle='--')
    axs[0].plot(ground_truths[:, 0], label='Ground Truth - Speed', color='red', marker='x', linestyle='-.')
    axs[0].set_title('Speed Comparison')
    axs[0].set_xlabel('Sample Index')
    axs[0].set_ylabel('Speed')
    axs[0].legend()

    axs[1].plot(model_outputs[:, 1], label='Model Output - Angle', color='green', marker='o', linestyle='--')
    axs[1].plot(ground_truths[:, 1], label='Ground Truth - Angle', color='orange', marker='x', linestyle='-.')
    axs[1].set_title('Angle Comparison')
    axs[1].set_xlabel('Sample Index')
    axs[1].set_ylabel('Angle')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f'/home/imagelab/zys/idp3_comparison_{idx}.png')

# allow for detecting segmentation fault
# import faulthandler
# faulthandler.enable()
# cprint("[fault handler enabled]", "cyan")

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    policy = workspace.get_model()
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    for i in range(0, len(dataset), 100):
        input_data = dataset.__getitem__(i)
        device = torch.device(cfg.training.device)
        batch = dict_apply(input_data, lambda x: x.unsqueeze(0).to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x)
        obs_dict = batch['obs']
        result = policy.predict_action(obs_dict)
        plot_result(result['action'][0].cpu().detach().numpy(), batch['action'][0][1:].cpu().detach().numpy(), i)

if __name__ == "__main__":
    main()

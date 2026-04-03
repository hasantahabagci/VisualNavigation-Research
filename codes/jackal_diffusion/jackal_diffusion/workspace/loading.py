from __future__ import annotations

import hydra
import torch
import dill

from jackal_diffusion.workspace.base_workspace import BaseWorkspace


def load_workspace_from_checkpoint(
    checkpoint: str,
    output_dir: str | None = None,
    device: str = "cpu",
) -> BaseWorkspace:
    payload = torch.load(checkpoint, map_location=device, pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    include_keys = [key for key in payload["pickles"].keys() if key != "_output_dir"]
    workspace.load_payload(payload, include_keys=include_keys)
    return workspace


def load_policy_from_checkpoint(
    checkpoint: str,
    output_dir: str | None = None,
    device: str = "cpu",
):
    workspace = load_workspace_from_checkpoint(
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=device,
    )
    policy = workspace.model
    policy.to(device)
    policy.eval()
    return workspace, policy

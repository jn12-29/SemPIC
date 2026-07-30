import os
import random
import tempfile
from typing import Any

import torch

from ..packet_wrapper import PacketWrapper
from .train import TrainConfig, TrainTarget


def checkpoint_path(train_config: TrainConfig) -> str:
    return os.path.join(train_config["output_dir"], "checkpoint.pt")


def _lora_state(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if "lora_" in name
    }


def _packet_state(packet_wrapper: PacketWrapper | None) -> dict[str, torch.Tensor] | None:
    if packet_wrapper is None:
        return None
    return {
        "header": packet_wrapper.header.detach().cpu().clone().requires_grad_(
            packet_wrapper.header.requires_grad
        ),
        "trailer": packet_wrapper.trailer.detach().cpu().clone().requires_grad_(
            packet_wrapper.trailer.requires_grad
        ),
    }


def save_training_checkpoint(
    path: str,
    *,
    next_epoch: int,
    epoch_indices: list[int],
    model: Any,
    packet_wrapper: PacketWrapper | None,
    optimizers: dict[TrainTarget, torch.optim.Optimizer],
    schedulers: dict[TrainTarget, torch.optim.lr_scheduler.LRScheduler],
) -> None:
    output_dir = os.path.dirname(path)
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "next_epoch": next_epoch,
        "epoch_indices": epoch_indices,
        "lora": _lora_state(model) or None,
        "packet_wrapper": _packet_state(packet_wrapper),
        "optimizers": {target: optimizer.state_dict() for target, optimizer in optimizers.items()},
        "schedulers": {target: scheduler.state_dict() for target, scheduler in schedulers.items()},
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir,
            prefix="checkpoint.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name
        torch.save(state, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def load_training_checkpoint(
    path: str,
    *,
    model: Any,
    packet_wrapper: PacketWrapper | None,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    lora_state = state.get("lora")
    if lora_state is not None:
        named_parameters = dict(model.named_parameters())
        with torch.no_grad():
            for name, saved_param in lora_state.items():
                named_parameters[name].copy_(saved_param)
    packet_state = state.get("packet_wrapper")
    if packet_state is not None:
        if packet_wrapper is None:
            raise ValueError("Checkpoint contains PacketWrapper state but it is not enabled.")
        packet_wrapper.load_state_dict({
            "header": packet_state["header"],
            "trailer": packet_state["trailer"],
            "train_config": {},
        })
    return state


def restore_training_state(
    state: dict[str, Any],
    *,
    optimizers: dict[TrainTarget, torch.optim.Optimizer],
    schedulers: dict[TrainTarget, torch.optim.lr_scheduler.LRScheduler],
) -> tuple[int, list[int]]:
    for target, optimizer in optimizers.items():
        optimizer.load_state_dict(state["optimizers"][target])
        schedulers[target].load_state_dict(state["schedulers"][target])
    random.setstate(state["python_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    cuda_rng_states = state.get("cuda_rng_states")
    if cuda_rng_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_states)
    return int(state["next_epoch"]), list(state["epoch_indices"])

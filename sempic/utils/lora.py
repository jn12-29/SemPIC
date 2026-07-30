from contextlib import contextmanager
from typing import Any, Iterator

import torch


def get_model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    return next(model.parameters()).device


def get_causal_lm_body(model: Any) -> Any:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    body = getattr(base_model, "model", None)
    if body is not None:
        return body

    base_model_attr = getattr(base_model, "base_model", None)
    body = getattr(base_model_attr, "model", None)
    if body is not None:
        return body

    raise AttributeError("Could not locate the causal LM body on the model.")


def set_lora_trainable_only(model: Any) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    if not any(param.requires_grad for param in model.parameters()):
        raise ValueError("No LoRA parameters are trainable. Check target_modules and adapter loading.")


def _param_requires_grad_states(model: Any) -> tuple[tuple[Any, bool], ...]:
    return tuple((param, param.requires_grad) for param in model.parameters())


def _restore_param_requires_grad_states(states: tuple[tuple[Any, bool], ...]) -> None:
    for param, requires_grad in states:
        param.requires_grad = requires_grad


def _adapter_layer_states(model: Any) -> tuple[tuple[Any, bool], ...]:
    states = []
    for module in model.modules() if hasattr(model, "modules") else ():
        if hasattr(module, "_disable_adapters"):
            states.append((module, bool(getattr(module, "disable_adapters", module._disable_adapters))))
    return tuple(states)


def _restore_adapter_layer_states(states: tuple[tuple[Any, bool], ...]) -> None:
    for module, was_disabled in states:
        if hasattr(module, "enable_adapters"):
            module.enable_adapters(not was_disabled)
        else:
            module._disable_adapters = was_disabled


def enable_lora_adapters(model: Any, adapter_name: str|None = None) -> None:
    param_states = _param_requires_grad_states(model)
    if adapter_name is not None and hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)
    layer_api_used = False
    try:
        if hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()
            layer_api_used = True
        if hasattr(model, "enable_adapters"):
            try:
                model.enable_adapters()
            except ValueError:
                if not layer_api_used:
                    raise
    finally:
        _restore_param_requires_grad_states(param_states)


def disable_lora_adapters(model: Any) -> None:
    param_states = _param_requires_grad_states(model)
    layer_api_used = False
    try:
        if hasattr(model, "disable_adapter_layers"):
            model.disable_adapter_layers()
            layer_api_used = True
        if hasattr(model, "disable_adapters"):
            try:
                model.disable_adapters()
            except ValueError:
                if not layer_api_used:
                    raise
    finally:
        _restore_param_requires_grad_states(param_states)


@contextmanager
def lora_adapters_enabled(model: Any, adapter_name: str|None = None) -> Iterator[None]:
    states = _adapter_layer_states(model)
    param_states = _param_requires_grad_states(model)
    enable_lora_adapters(model, adapter_name=adapter_name)
    try:
        yield
    finally:
        _restore_adapter_layer_states(states)
        _restore_param_requires_grad_states(param_states)


@contextmanager
def lora_adapters_disabled(model: Any) -> Iterator[None]:
    states = _adapter_layer_states(model)
    param_states = _param_requires_grad_states(model)
    disable_lora_adapters(model)
    try:
        yield
    finally:
        _restore_adapter_layer_states(states)
        _restore_param_requires_grad_states(param_states)

import argparse
import logging
import sys
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import TypedDict
import torch
from torch import Tensor
from torch.nn import Embedding
from sempic.packet_wrapper import PacketWrapper, WrapperStateDict
from sempic.utils.generate import TokenizerType
from sempic.model import SupportedModel
from sempic.utils.config import gather_config_files, load_config_file
from sempic.utils.runtime import RuntimeContext, apply_runtime_overrides
from sempic.utils.run_storage import allocate_run_dir

LOGGER = logging.getLogger("sempic.run_build_packet")

class ModelConfig(TypedDict):
    model_path: str
    dtype: str
    device: str


class TrainConfig(TypedDict):
    header: list[str|list[int]]
    trailer: list[str|list[int]]
    dtype: str|None
    model: ModelConfig
    save_path: str
    file_name: str


def load_train_config(config: dict[str, Any]) -> TrainConfig:
    model_config_dict = config["model"]
    model_config = ModelConfig(
        model_path=model_config_dict["model_path"],
        dtype=model_config_dict["dtype"],
        device=model_config_dict["device"]
    )
    train_config = TrainConfig(
        header=config["header"],
        trailer=config["trailer"],
        dtype=config.get("dtype", None),
        model=model_config,
        save_path=config["save_path"],
        file_name=config["file_name"]
    )

    return train_config


class TrainCache(TypedDict):
    model: dict[tuple[str, str, str], SupportedModel]
    tokenizer: dict[tuple[str, str, str], TokenizerType]


def convert_adapter(
    adapter: list[str|list[int]],
    tokenizer: TokenizerType,
    embed_layer: Embedding
) -> Tensor:
    """
    Convert the adapter tokens to embeddings
    if the adapter is a string, it will be tokenized and converted to embeddings
    if the adapter is a list of integers, it will be directly converted to embeddings

    Args:
        adapter (str|list[int]): The adapter tokens
        tokenizer (TokenizerType): The tokenizer to use for tokenization
        embed_layer (Embedding): The embedding layer to use for conversion
    Returns:
        Tensor: The converted adapter embeddings
    """
    token_list: list[Tensor] = []

    for item in adapter:
        if isinstance(item, str):
            tokens = tokenizer(item, return_tensors="pt", add_special_tokens=False).input_ids
            assert isinstance(tokens, Tensor)
            LOGGER.info("Tokenized %r to %s, with length %d", item, tokens, tokens.shape[1])
        elif isinstance(item, list):
            tokens = torch.tensor([item], dtype=torch.long)
        else:
            raise ValueError("Adapter items must be a string or a list of integers")
        
        token_list.append(tokens)

    tokens = torch.cat(token_list, dim=1)
    LOGGER.info("Stacked tokens shape: %s", tokens.shape)
    tokens = tokens.to(embed_layer.weight.device)
    embeddings = embed_layer(tokens)
    return embeddings



def run_one_train_config(
    train_config: TrainConfig,
    train_cache: TrainCache
) -> None:
    model_config = train_config["model"]
    model_key = (
        model_config["model_path"],
        model_config["dtype"],
        model_config["device"]
    )
    if model_key in train_cache["model"]:
        model: Any = train_cache["model"][model_key]
        assert isinstance(model, SupportedModel)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_config["model_path"],
            dtype=model_config["dtype"],
            device_map=model_config["device"],
            low_cpu_mem_usage=True
        )
        assert isinstance(model, SupportedModel)
        train_cache["model"][model_key] = model
    
    if model_key in train_cache["tokenizer"]:
        tokenizer = train_cache["tokenizer"][model_key]
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model_path"],
            use_fast=True
        )
        train_cache["tokenizer"][model_key] = tokenizer

    model_embed_dim = model.config.hidden_size
    assert model_embed_dim is not None

    header = train_config["header"]
    trailer = train_config["trailer"]

    embed_layer = model.model.embed_tokens

    device = torch.device(model_config["device"])
    header_embeddings = convert_adapter(header, tokenizer, embed_layer).to(device)
    trailer_embeddings = convert_adapter(trailer, tokenizer, embed_layer).to(device)

    packet_wrapper_state_dict: WrapperStateDict = {
        "header": header_embeddings,
        "trailer": trailer_embeddings,
        "train_config": {}
    }
    packet_wrapper = PacketWrapper.from_state_dict(packet_wrapper_state_dict)

    save_path = train_config["save_path"]
    file_name = train_config["file_name"]

    torch.save(packet_wrapper.state_dict(), f"{save_path}/{file_name}")
    LOGGER.info("Packet saved to %s/%s", save_path, file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a packet from a config file")
    parser.add_argument(
        "config_files_or_paths",
        type=str,
        nargs="+",
        help="Path to the training configuration file (JSON format)."
    )
    args = parser.parse_args()


    config_files_or_paths: list[str] = args.config_files_or_paths
    assert isinstance(config_files_or_paths, list)

    all_config_files: list[str] = []
    for file_or_path in config_files_or_paths:
        config_files = gather_config_files(file_or_path, pattern=r".*\.json$")
        all_config_files.extend(config_files)

    if not all_config_files:
        run_dir = allocate_run_dir("./logs/run_build_packet", None)
        with RuntimeContext(
            entrypoint="run_build_packet",
            run_dir=run_dir,
            config_file=None,
            resolved_config=None,
            config_snapshot_name=None,
            cli_args=vars(args).copy(),
        ) as runtime_context:
            runtime_context.logger.info("Loaded 0 packet build configurations.")
        sys.exit(0)

    train_cache: TrainCache = {
        "model": {},
        "tokenizer": {}
    }

    for config_file in all_config_files:
        config = apply_runtime_overrides(
            load_config_file(config_file, default_config_file="_default.json")
        )
        train_config = load_train_config(config)
        run_dir = allocate_run_dir("./logs/run_build_packet", None)
        with RuntimeContext(
            entrypoint="run_build_packet",
            run_dir=run_dir,
            config_file=config_file,
            resolved_config=config,
            config_snapshot_name="build_packet_config.json",
            cli_args=vars(args).copy(),
        ) as runtime_context:
            runtime_context.logger.info("Running packet build for config: %s", config_file)
            run_one_train_config(train_config, train_cache)

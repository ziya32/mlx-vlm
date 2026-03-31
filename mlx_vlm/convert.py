import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Set, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path
from mlx_lm.utils import dequantize_model, quantize_model

from .utils import (
    MODEL_CONVERSION_DTYPES,
    create_model_card,
    fetch_from_hub,
    get_model_path,
    save_config,
    save_weights,
    skip_multimodal_module,
    upload_to_hub,
)

QUANT_RECIPES = [
    "mixed_2_6",
    "mixed_3_4",
    "mixed_3_5",
    "mixed_3_6",
    "mixed_3_8",
    "mixed_4_6",
    "mixed_4_8",
]


def mixed_quant_predicate_builder(
    recipe: str, model: nn.Module
) -> Callable[[str, nn.Module], Union[bool, dict]]:
    group_size = 64

    recipe_config = {
        "mixed_2_6": (2, 6),
        "mixed_3_4": (3, 4),
        "mixed_3_5": (3, 5),
        "mixed_3_6": (3, 6),
        "mixed_3_8": (3, 8),
        "mixed_4_6": (4, 6),
        "mixed_4_8": (4, 8),
    }

    if recipe not in recipe_config:
        raise ValueError(f"Invalid quant recipe {recipe}")

    low_bits, high_bits = recipe_config[recipe]

    down_keys = [k for k, _ in model.named_modules() if "down_proj" in k]
    if len(down_keys) == 0:
        raise ValueError("Model does not have expected keys for mixed quant.")

    # Look for the layer index location in the path:
    for layer_location, k in enumerate(down_keys[0].split(".")):
        if k.isdigit():
            break
    num_layers = len(model.layers)

    def mixed_quant_predicate(
        path: str,
        module: nn.Module,
    ) -> Union[bool, dict]:
        """Implements mixed quantization predicates with similar choices to, for example, llama.cpp's Q4_K_M.
        Ref: https://github.com/ggerganov/llama.cpp/blob/917786f43d0f29b7c77a0c56767c0fa4df68b1c5/src/llama.cpp#L5265
        By Alex Barron: https://gist.github.com/barronalex/84addb8078be21969f1690c1454855f3
        """

        if skip_multimodal_module(path):
            return False
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[1] % group_size != 0:
            return False

        path_parts = path.split(".")
        index = 0

        if len(path_parts) > layer_location:
            element = path_parts[layer_location]
            if element.isdigit():
                index = int(element)

        use_more_bits = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )

        if use_more_bits and ("v_proj" in path or "down_proj" in path):
            return {"group_size": group_size, "bits": high_bits}

        if "lm_head" in path or "embed_tokens" in path:
            return {"group_size": group_size, "bits": high_bits}

        return {"group_size": group_size, "bits": low_bits}

    return mixed_quant_predicate


def _build_skip_set(modules_to_not_convert: List[str]) -> Set[str]:
    """Build a set of normalized module paths for skip matching.

    HuggingFace paths use a ``model.`` prefix (the HF wrapper) and may omit
    intermediate containers that exist in the MLX model tree (e.g. the inner
    ``.model`` inside ``LanguageModel``).  We generate several normalised
    variants of every entry so that a simple set/endswith lookup against the
    MLX ``named_modules()`` paths will match correctly.
    """
    skip: Set[str] = set()
    for path in modules_to_not_convert:
        skip.add(path)
        # Strip leading "model." prefix (HF wrapper)
        if path.startswith("model."):
            stripped = path[len("model."):]
            skip.add(stripped)
            # Insert ".model" after the first component to account for the
            # inner model container in MLX LanguageModel classes.
            # e.g. "language_model.layers.0.mlp.gate"
            #    → "language_model.model.layers.0.mlp.gate"
            parts = stripped.split(".", 1)
            if len(parts) == 2:
                skip.add(f"{parts[0]}.model.{parts[1]}")
    return skip


def _should_skip_module(mlx_path: str, skip_set: Set[str]) -> bool:
    """Check whether *mlx_path* matches any entry in *skip_set*."""
    if mlx_path in skip_set:
        return True
    for skip_path in skip_set:
        if mlx_path.endswith("." + skip_path):
            return True
    return False


def load_modules_to_not_convert(quant_config_path: str) -> List[str]:
    """Load modules_to_not_convert from a config.json that contains a
    ``quantization_config`` section."""
    path = Path(quant_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Quant config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    qcfg = cfg.get("quantization_config", {})
    modules = qcfg.get("modules_to_not_convert", [])
    if not modules:
        raise ValueError(
            f"No modules_to_not_convert found in {path}"
        )
    return modules


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    q_mode: str = "affine",
    dtype: Optional[str] = None,
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    trust_remote_code: bool = True,
    quant_predicate: Optional[str] = None,
    quant_config: Optional[str] = None,
):
    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)
    model, config, processor = fetch_from_hub(
        model_path, lazy=True, trust_remote_code=trust_remote_code
    )

    model_quant_predicate = getattr(model, "quant_predicate", None)

    # Load modules to skip from an external quantization config
    skip_set: Optional[Set[str]] = None
    if quant_config is not None:
        modules_to_not_convert = load_modules_to_not_convert(quant_config)
        skip_set = _build_skip_set(modules_to_not_convert)
        print(f"[INFO] Loaded {len(modules_to_not_convert)} modules to skip from {quant_config}")

    def base_quant_predicate(path, module):
        if skip_multimodal_module(path):
            return False
        if skip_set is not None and _should_skip_module(path, skip_set):
            return False
        if model_quant_predicate is not None:
            return model_quant_predicate(path, module)
        return True

    if isinstance(quant_predicate, str):
        quant_predicate = mixed_quant_predicate_builder(quant_predicate, model)

    quant_predicate = quant_predicate or base_quant_predicate

    if dtype is None:
        dtype = config.get("torch_dtype", None)
    if dtype is None and (text_config := config.get("text_config", None)):
        dtype = text_config.get("dtype", None)
    if dtype in MODEL_CONVERSION_DTYPES:
        print("[INFO] Using dtype:", dtype)
        dtype = getattr(mx, dtype)
        cast_predicate = getattr(model, "cast_predicate", lambda _: True)

        def set_dtype(k, v):
            if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating):
                return v.astype(dtype)
            else:
                return v

        model.update(tree_map_with_path(set_dtype, model.parameters()))

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        config.setdefault("vision_config", {})
        model, config = quantize_model(
            model,
            config,
            q_group_size,
            q_bits,
            mode=q_mode,
            quant_predicate=quant_predicate,
        )

    if dequantize:
        print("[INFO] Dequantizing")
        model = dequantize_model(model)

    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    save_weights(mlx_path, model, donate_weights=True)

    # Copy Python and JSON files from the model path to the MLX path
    for pattern in ["*.py", "*.json"]:
        files = glob.glob(str(model_path / pattern))
        for file in files:
            # Skip the index file - save_weights() already generated the correct one
            if Path(file).name == "model.safetensors.index.json":
                continue
            shutil.copy(file, mlx_path)

    # Copy folders from the model path to the MLX path
    for item in model_path.iterdir():
        if item.is_dir():
            dest = mlx_path / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)

    processor.save_pretrained(mlx_path)

    save_config(config, config_path=mlx_path / "config.json")

    hf_repo = None if Path(hf_path).exists() else hf_path
    create_model_card(mlx_path, hf_repo)

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo)


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )
    parser.add_argument(
        "--hf-path",
        "--model",
        type=str,
        help="Path to the model. This can be a local path or a Hugging Face Hub model identifier.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        help="Hugging Face revision (branch), when converting a model from the Hub.",
        default=None,
    )
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size",
        help="Group size for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-bits",
        help="Bits per weight for quantization.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--q-mode",
        help="The quantization mode.",
        type=str,
        choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
        default="affine",
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the parameter. Defaults to config.json's `torch_dtype` or the current model weights dtype",
        type=str,
        choices=MODEL_CONVERSION_DTYPES,
        default=None,
    )
    parser.add_argument(
        "--quant-predicate",
        help=f"Mixed-bit quantization recipe.",
        choices=QUANT_RECIPES,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--trust-remote-code",
        help="Trust remote code.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--quant-config",
        help="Path to a config.json containing quantization_config with "
        "modules_to_not_convert. Listed modules will be skipped during quantization.",
        type=str,
        default=None,
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_vlm.convert ...` directly is deprecated."
        " Use `mlx_vlm.convert ...` or `python -m mlx_vlm convert ...` instead."
    )
    main()

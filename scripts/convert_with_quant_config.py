#!/usr/bin/env python3
"""Convert a HuggingFace model to MLX format, skipping modules listed in
a quantization config's ``modules_to_not_convert``.

Example usage:

    python scripts/convert_with_quant_config.py \
        --hf-path /path/to/base/model \
        --mlx-path /path/to/output \
        --quant-config /path/to/fp8/config.json \
        --q-bits 4
"""

from mlx_vlm.convert import convert, configure_parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    main()

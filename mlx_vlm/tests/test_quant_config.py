import json
import tempfile
from pathlib import Path

import pytest

from mlx_vlm.convert import (
    _build_skip_set,
    _should_skip_module,
    load_modules_to_not_convert,
)


# ---------------------------------------------------------------------------
# _build_skip_set
# ---------------------------------------------------------------------------

class TestBuildSkipSet:
    def test_plain_path_kept(self):
        skip = _build_skip_set(["lm_head"])
        assert "lm_head" in skip

    def test_model_prefix_stripped(self):
        skip = _build_skip_set(["model.language_model.layers.0.mlp.gate"])
        assert "language_model.layers.0.mlp.gate" in skip

    def test_inner_model_inserted(self):
        """HF path 'model.language_model.layers.0.mlp.gate' should produce
        'language_model.model.layers.0.mlp.gate' (the MLX path)."""
        skip = _build_skip_set(["model.language_model.layers.0.mlp.gate"])
        assert "language_model.model.layers.0.mlp.gate" in skip

    def test_no_model_prefix_no_inner_model(self):
        """Paths without 'model.' prefix should not get inner '.model' inserted."""
        skip = _build_skip_set(["mtp.fc"])
        assert "mtp.fc" in skip
        assert "mtp.model.fc" not in skip

    def test_multiple_entries(self):
        skip = _build_skip_set([
            "lm_head",
            "model.language_model.embed_tokens",
            "model.language_model.layers.5.mlp.gate",
        ])
        # All originals present
        assert "lm_head" in skip
        assert "model.language_model.embed_tokens" in skip
        # Stripped variants
        assert "language_model.embed_tokens" in skip
        # Inner model variants
        assert "language_model.model.embed_tokens" in skip
        assert "language_model.model.layers.5.mlp.gate" in skip


# ---------------------------------------------------------------------------
# _should_skip_module
# ---------------------------------------------------------------------------

class TestShouldSkipModule:
    def test_exact_match(self):
        skip_set = {"language_model.model.layers.0.mlp.gate"}
        assert _should_skip_module("language_model.model.layers.0.mlp.gate", skip_set)

    def test_endswith_match(self):
        """MLX paths may have extra prefix; endswith should still match."""
        skip_set = {"lm_head"}
        assert _should_skip_module("language_model.lm_head", skip_set)

    def test_no_match(self):
        skip_set = {"lm_head", "language_model.model.layers.0.mlp.gate"}
        assert not _should_skip_module("language_model.model.layers.0.mlp.down_proj", skip_set)

    def test_partial_name_no_false_positive(self):
        """'gate' should NOT match 'shared_expert_gate' — the dot-separated
        boundary prevents partial name collisions."""
        skip_set = {"gate"}
        # "shared_expert_gate" ends with "gate" as a substring, but NOT ".gate"
        assert not _should_skip_module("model.layers.0.mlp.shared_expert_gate", skip_set)
        # But "model.layers.0.mlp.gate" does end with ".gate"
        assert _should_skip_module("model.layers.0.mlp.gate", skip_set)

    def test_full_hf_pipeline(self):
        """End-to-end: HF paths → _build_skip_set → _should_skip_module on MLX paths."""
        hf_paths = [
            "lm_head",
            "model.language_model.embed_tokens",
            "model.language_model.layers.0.mlp.gate",
            "model.language_model.layers.0.mlp.shared_expert_gate",
            "model.language_model.layers.10.linear_attn.conv1d",
            "mtp.fc",
        ]
        skip_set = _build_skip_set(hf_paths)

        # These MLX paths should be skipped
        assert _should_skip_module("language_model.lm_head", skip_set)
        assert _should_skip_module("language_model.model.embed_tokens", skip_set)
        assert _should_skip_module("language_model.model.layers.0.mlp.gate", skip_set)
        assert _should_skip_module("language_model.model.layers.0.mlp.shared_expert_gate", skip_set)
        assert _should_skip_module("language_model.model.layers.10.linear_attn.conv1d", skip_set)
        assert _should_skip_module("language_model.model.mtp.fc", skip_set)

        # These MLX paths should NOT be skipped
        assert not _should_skip_module("language_model.model.layers.0.mlp.down_proj", skip_set)
        assert not _should_skip_module("language_model.model.layers.1.mlp.down_proj", skip_set)
        assert not _should_skip_module("language_model.model.layers.0.self_attn.q_proj", skip_set)


# ---------------------------------------------------------------------------
# load_modules_to_not_convert
# ---------------------------------------------------------------------------

class TestLoadModulesToNotConvert:
    def test_loads_from_valid_config(self, tmp_path):
        config = {
            "model_type": "test",
            "quantization_config": {
                "quant_method": "fp8",
                "modules_to_not_convert": ["lm_head", "model.language_model.embed_tokens"],
            },
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(config))

        modules = load_modules_to_not_convert(str(cfg_path))
        assert modules == ["lm_head", "model.language_model.embed_tokens"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_modules_to_not_convert("/nonexistent/path/config.json")

    def test_no_quantization_config_raises(self, tmp_path):
        config = {"model_type": "test"}
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="No modules_to_not_convert"):
            load_modules_to_not_convert(str(cfg_path))

    def test_empty_modules_list_raises(self, tmp_path):
        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "modules_to_not_convert": [],
            },
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(config))

        with pytest.raises(ValueError, match="No modules_to_not_convert"):
            load_modules_to_not_convert(str(cfg_path))

    def test_loads_real_fp8_config(self):
        """Load the actual Qwen3.5-35B-A3B-FP8 config if available."""
        fp8_config = Path("/Volumes/Ext-HD/models-arch/alibaba/Qwen3.5-35B-A3B-FP8/config.json")
        if not fp8_config.exists():
            pytest.skip("FP8 config not available")

        modules = load_modules_to_not_convert(str(fp8_config))
        assert len(modules) > 0
        assert "lm_head" in modules
        assert "model.language_model.embed_tokens" in modules

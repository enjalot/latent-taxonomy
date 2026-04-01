"""SAE model loading and triage utilities."""

import torch
from dataclasses import dataclass
from pipeline.config import PipelineConfig


def load_sae_model(config: PipelineConfig):
    """Load SAE model from hub, local path, or modal volume.

    Returns the SAE model object with decoder weights accessible via
    model.W_dec (shape: [n_features, d_in]).
    """
    if config.model_source == "hub":
        from huggingface_hub import hf_hub_download
        import json
        import os

        # Download config and model files
        config_path = hf_hub_download(
            repo_id=config.model_repo,
            filename=f"{config.model_name}/cfg.json",
        )
        model_path = hf_hub_download(
            repo_id=config.model_repo,
            filename=f"{config.model_name}/sae_weights.safetensors",
        )

        with open(config_path) as f:
            cfg = json.load(f)

        from safetensors.torch import load_file
        state_dict = load_file(model_path)

        return SAEModel(state_dict, cfg)

    elif config.model_source == "local":
        import json
        import os

        base = config.model_path or "."
        config_path = os.path.join(base, config.model_name, "cfg.json")
        model_path = os.path.join(base, config.model_name, "sae_weights.safetensors")

        with open(config_path) as f:
            cfg = json.load(f)

        from safetensors.torch import load_file
        state_dict = load_file(model_path)

        return SAEModel(state_dict, cfg)

    else:
        raise ValueError(f"Unknown model source: {config.model_source}")


class SAEModel:
    """Minimal SAE model wrapper for accessing decoder weights."""

    def __init__(self, state_dict: dict, cfg: dict):
        self.cfg = cfg
        self.state_dict = state_dict
        # Decoder weight matrix: [n_features, d_in]
        if "W_dec" in state_dict:
            self.W_dec = state_dict["W_dec"]
        elif "decoder.weight" in state_dict:
            self.W_dec = state_dict["decoder.weight"]
        else:
            # Try to find any decoder weight
            for key in state_dict:
                if "dec" in key.lower():
                    self.W_dec = state_dict[key]
                    break
            else:
                raise ValueError(
                    f"Could not find decoder weights in state dict. Keys: {list(state_dict.keys())}"
                )

    @property
    def n_features(self) -> int:
        return self.W_dec.shape[0]

    @property
    def d_in(self) -> int:
        return self.W_dec.shape[1]

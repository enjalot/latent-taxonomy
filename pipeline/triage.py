"""SAE model loading utilities."""

import os

from pipeline.config import PipelineConfig


def load_sae_model(config: PipelineConfig):
    """Load SAE model from hub or local path using latentsae.

    Returns a latentsae.Sae object with:
    - model.W_dec: decoder weights [n_features, d_in]
    - model.encode(x): encode embeddings to sparse features
    - model.num_latents: number of features
    - model.d_in: input dimension
    """
    from latentsae.sae import Sae

    if config.model_source == "hub":
        print(f"Loading SAE from hub: {config.model_repo} / {config.model_name}")
        return Sae.load_from_hub(config.model_repo, config.model_name)

    elif config.model_source == "local":
        path = os.path.expanduser(config.model_path)
        print(f"Loading SAE from local: {path}")
        return Sae.load_from_disk(path)

    elif config.model_source == "modal":
        import subprocess
        import tempfile

        local_dir = os.path.join(
            tempfile.gettempdir(), "latent-taxonomy-modal",
            config.modal_volume, config.modal_path or ""
        )
        os.makedirs(local_dir, exist_ok=True)

        cfg_path = os.path.join(local_dir, "cfg.json")
        sae_path = os.path.join(local_dir, "sae.safetensors")
        if not (os.path.exists(cfg_path) and os.path.exists(sae_path)):
            print(f"Downloading from Modal volume '{config.modal_volume}'...")
            for fname in ["cfg.json", "sae.safetensors"]:
                remote = f"{config.modal_path}/{fname}" if config.modal_path else fname
                subprocess.run(
                    ["modal", "volume", "get", config.modal_volume, remote, os.path.join(local_dir, fname)],
                    check=True,
                )

        return Sae.load_from_disk(local_dir)

    else:
        raise ValueError(f"Unknown model source: {config.model_source}")

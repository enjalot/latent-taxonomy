from dataclasses import dataclass


@dataclass
class PipelineConfig:
    model_source: str = "hub"
    model_repo: str = "enjalot/sae-nomic-text-v1.5-FineWeb-edu-100BT"
    model_name: str = "64_32"
    model_path: str = None
    modal_volume: str = "checkpoints"
    modal_path: str = None
    samples_path: str = "notebooks/data/top10_64_32.parquet"
    output_dir: str = "pipeline/output"
    min_activation: float = 0.01
    min_samples: int = 5
    duplicate_threshold: float = 0.95
    hdbscan_min_cluster_size: int = 5

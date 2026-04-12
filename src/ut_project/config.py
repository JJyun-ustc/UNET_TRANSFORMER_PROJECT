from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT.parent / "testcode" / "image"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
NOISE_PRESETS = {
    "small": (0.0, 0.05),
    "medium": (0.05, 0.15),
    "large": (0.15, 0.30),
}
CHECKPOINT_DIRS = {
    "small": DEFAULT_CHECKPOINT_ROOT / "small",
    "medium": DEFAULT_CHECKPOINT_ROOT / "medium",
    "large": DEFAULT_CHECKPOINT_ROOT / "large",
    "custom": DEFAULT_CHECKPOINT_ROOT / "custom",
}


@dataclass
class TrainConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_ROOT / "custom"
    batch_size: int = 8
    num_workers: int = 0
    patch_size: int = 256
    epochs: int = 12
    learning_rate: float = 1e-3
    noise_min: float = 0.0
    noise_max: float = 0.2
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32
    embed_dim: int = 128
    num_heads: int = 4
    transformer_layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.0
    seed: int = 42


@dataclass
class InferConfig:
    checkpoint_path: Path
    input_path: Path
    output_path: Path | None = None
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32
    embed_dim: int = 128
    num_heads: int = 4
    transformer_layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.0

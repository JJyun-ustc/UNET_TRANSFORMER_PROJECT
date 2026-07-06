from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "image"
DEFAULT_TRAIN_ROOT = DEFAULT_DATA_ROOT / "train_data" / "Data"
DEFAULT_VAL_NOISY_MAT = DEFAULT_DATA_ROOT / "ValidationNoisyBlocksSrgb.mat"
DEFAULT_VAL_GT_MAT = DEFAULT_DATA_ROOT / "ValidationGtBlocksSrgb.mat"
DEFAULT_BENCHMARK_ROOT = DEFAULT_DATA_ROOT / "SIDD_Benchmark_Data"
DEFAULT_BENCHMARK_BLOCKS = DEFAULT_DATA_ROOT / "SIDD_Benchmark_Code_v1.2" / "BenchmarkBlocks32.mat"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIRS = {
    "sidd": DEFAULT_CHECKPOINT_ROOT / "sidd",
}


@dataclass
class TrainConfig:
    train_root: Path = DEFAULT_TRAIN_ROOT
    val_noisy_mat: Path = DEFAULT_VAL_NOISY_MAT
    val_gt_mat: Path = DEFAULT_VAL_GT_MAT
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_ROOT / "sidd"
    batch_size: int = 8
    num_workers: int = 0
    patch_size: int = 256
    epochs: int = 12
    learning_rate: float = 1e-3
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

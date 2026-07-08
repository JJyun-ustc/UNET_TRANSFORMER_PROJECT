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

DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 8
DEFAULT_VAL_BATCH_SIZE = 8
DEFAULT_PATCH_SIZE = 256
DEFAULT_NUM_WORKERS = 0
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_PAIRS_PER_IMAGE = 1
DEFAULT_SEED = 42

DEFAULT_IN_CHANNELS = 3
DEFAULT_OUT_CHANNELS = 3
DEFAULT_BASE_CHANNELS = 32
DEFAULT_EMBED_DIM = 128
DEFAULT_NUM_HEADS = 4
DEFAULT_TRANSFORMER_LAYERS = 4
DEFAULT_FF_DIM = 256
DEFAULT_DROPOUT = 0.0

DEFAULT_NUM_SAMPLES = 3


@dataclass
class TrainConfig:
    train_root: Path = DEFAULT_TRAIN_ROOT
    val_noisy_mat: Path = DEFAULT_VAL_NOISY_MAT
    val_gt_mat: Path = DEFAULT_VAL_GT_MAT
    checkpoint_dir: Path = CHECKPOINT_DIRS["sidd"]
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    val_batch_size: int = DEFAULT_VAL_BATCH_SIZE
    patch_size: int = DEFAULT_PATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    learning_rate: float = DEFAULT_LEARNING_RATE
    pairs_per_image: int = DEFAULT_PAIRS_PER_IMAGE
    seed: int = DEFAULT_SEED
    in_channels: int = DEFAULT_IN_CHANNELS
    out_channels: int = DEFAULT_OUT_CHANNELS
    base_channels: int = DEFAULT_BASE_CHANNELS
    embed_dim: int = DEFAULT_EMBED_DIM
    num_heads: int = DEFAULT_NUM_HEADS
    transformer_layers: int = DEFAULT_TRANSFORMER_LAYERS
    ff_dim: int = DEFAULT_FF_DIM
    dropout: float = DEFAULT_DROPOUT


@dataclass
class InferConfig:
    checkpoint_path: Path | None = None
    input_path: Path | None = None
    target_path: Path | None = None
    output_path: Path | None = None
    val_noisy_mat: Path = DEFAULT_VAL_NOISY_MAT
    val_gt_mat: Path = DEFAULT_VAL_GT_MAT
    num_samples: int = DEFAULT_NUM_SAMPLES
    validate_all: bool = False
    val_batch_size: int = DEFAULT_VAL_BATCH_SIZE
    metrics_output: Path | None = None
    in_channels: int = DEFAULT_IN_CHANNELS
    out_channels: int = DEFAULT_OUT_CHANNELS
    base_channels: int = DEFAULT_BASE_CHANNELS
    embed_dim: int = DEFAULT_EMBED_DIM
    num_heads: int = DEFAULT_NUM_HEADS
    transformer_layers: int = DEFAULT_TRANSFORMER_LAYERS
    ff_dim: int = DEFAULT_FF_DIM
    dropout: float = DEFAULT_DROPOUT

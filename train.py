import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ut_project.config import CHECKPOINT_DIRS, TrainConfig
from ut_project.data import build_sidd_train_dataloader, build_sidd_validation_dataloader
from ut_project.engine import Trainer
from ut_project.engine.trainer import set_seed
from ut_project.models import UNetTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a UNet + Transformer denoising model on SIDD Medium sRGB.")
    parser.add_argument("--train-root", type=Path, default=None, help="SIDD Medium train_data/Data directory")
    parser.add_argument("--val-noisy-mat", type=Path, default=None, help="ValidationNoisyBlocksSrgb.mat path")
    parser.add_argument("--val-gt-mat", type=Path, default=None, help="ValidationGtBlocksSrgb.mat path")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--pairs-per-image",
        type=int,
        default=1,
        help="How many random patches to draw from each SIDD pair per epoch.",
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig()

    if args.train_root is not None:
        config.train_root = args.train_root
    if args.val_noisy_mat is not None:
        config.val_noisy_mat = args.val_noisy_mat
    if args.val_gt_mat is not None:
        config.val_gt_mat = args.val_gt_mat

    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.patch_size = args.patch_size
    config.num_workers = args.num_workers
    config.learning_rate = args.lr
    config.base_channels = args.base_channels
    config.embed_dim = args.embed_dim
    config.num_heads = args.num_heads
    config.transformer_layers = args.transformer_layers
    config.ff_dim = args.ff_dim
    config.dropout = args.dropout
    config.seed = args.seed
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    else:
        config.checkpoint_dir = CHECKPOINT_DIRS["sidd"]

    print(f"training root: {config.train_root}")
    print(f"validation noisy mat: {config.val_noisy_mat}")
    print(f"validation gt mat: {config.val_gt_mat}")
    print(f"checkpoint dir: {config.checkpoint_dir}")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    train_loader = build_sidd_train_dataloader(
        root_dir=config.train_root,
        batch_size=config.batch_size,
        patch_size=config.patch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pairs_per_image=args.pairs_per_image,
    )
    test_loader = build_sidd_validation_dataloader(
        noisy_mat=config.val_noisy_mat,
        gt_mat=config.val_gt_mat,
        batch_size=args.val_batch_size,
        num_workers=config.num_workers,
    )

    model = UNetTransformer(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        transformer_layers=config.transformer_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
    ).to(device)

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()

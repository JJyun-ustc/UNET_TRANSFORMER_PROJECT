import argparse
import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ut_project.config import InferConfig
from ut_project.data import SIDDValidationBlocksDataset
from ut_project.engine.trainer import calculate_psnr
from ut_project.models import UNetTransformer


def parse_args():
    config = InferConfig()
    parser = argparse.ArgumentParser(description="Run inference with a SIDD-trained UNet + Transformer denoiser.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=config.input_path, help="Noisy sRGB image path")
    parser.add_argument("--target", type=Path, default=config.target_path, help="Optional clean/GT sRGB image path for PSNR")
    parser.add_argument("--output", type=Path, default=config.output_path)
    parser.add_argument("--val-noisy-mat", type=Path, default=config.val_noisy_mat)
    parser.add_argument("--val-gt-mat", type=Path, default=config.val_gt_mat)
    parser.add_argument("--num-samples", type=int, default=config.num_samples)
    parser.add_argument("--validate-all", action="store_true", help="Run the full SIDD validation set and print metrics.")
    parser.add_argument("--val-batch-size", type=int, default=config.val_batch_size)
    parser.add_argument("--metrics-output", type=Path, default=config.metrics_output, help="Optional CSV path for per-block metrics.")
    parser.add_argument("--base-channels", type=int, default=config.base_channels)
    parser.add_argument("--embed-dim", type=int, default=config.embed_dim)
    parser.add_argument("--num-heads", type=int, default=config.num_heads)
    parser.add_argument("--transformer-layers", type=int, default=config.transformer_layers)
    parser.add_argument("--ff-dim", type=int, default=config.ff_dim)
    parser.add_argument("--dropout", type=float, default=config.dropout)
    return parser.parse_args()


def load_image_tensor(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    x = transforms.ToTensor()(image).unsqueeze(0)
    return image, x


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    return transforms.ToPILImage()(x.squeeze(0).clamp(0.0, 1.0).cpu())


def denoise_tensor(model, noisy: torch.Tensor):
    pred_noise = model(noisy)
    return torch.clamp(noisy - pred_noise, 0.0, 1.0)


def calculate_mse(img1: torch.Tensor, img2: torch.Tensor):
    return torch.mean((img1 - img2) ** 2).item()


def validate_all_blocks(model, device, noisy_mat: Path, gt_mat: Path, batch_size: int, metrics_output: Path | None):
    dataset = SIDDValidationBlocksDataset(noisy_mat=noisy_mat, gt_mat=gt_mat)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    rows = []
    total_psnr = 0.0
    total_mse = 0.0
    min_psnr = float("inf")
    max_psnr = float("-inf")
    block_count = 0

    with torch.no_grad():
        for batch_index, (noisy, clean) in enumerate(loader):
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            denoised = denoise_tensor(model, noisy)

            for item_index, (clean_image, denoised_image) in enumerate(zip(clean, denoised)):
                block_index = batch_index * batch_size + item_index
                psnr = calculate_psnr(clean_image.cpu(), denoised_image.cpu())
                mse = calculate_mse(clean_image.cpu(), denoised_image.cpu())
                rows.append(
                    {
                        "block_index": block_index,
                        "psnr_db": f"{psnr:.6f}",
                        "mse": f"{mse:.8f}",
                    }
                )
                total_psnr += psnr
                total_mse += mse
                min_psnr = min(min_psnr, psnr)
                max_psnr = max(max_psnr, psnr)
                block_count += 1

    avg_psnr = total_psnr / max(block_count, 1)
    avg_mse = total_mse / max(block_count, 1)
    print(f"validated blocks: {block_count}")
    print(f"avg_psnr: {avg_psnr:.2f} dB")
    print(f"min_psnr: {min_psnr:.2f} dB")
    print(f"max_psnr: {max_psnr:.2f} dB")
    print(f"avg_mse: {avg_mse:.8f}")

    if metrics_output is not None:
        metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with metrics_output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["block_index", "psnr_db", "mse"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved validation metrics to: {metrics_output}")


def show_validation_samples(model, device, noisy_mat: Path, gt_mat: Path, num_samples: int):
    dataset = SIDDValidationBlocksDataset(noisy_mat=noisy_mat, gt_mat=gt_mat)
    sample_count = min(num_samples, len(dataset))
    sample_indices = random.sample(range(len(dataset)), k=sample_count)

    plt.figure(figsize=(12, 4 * sample_count))
    metrics_rows = []

    with torch.no_grad():
        for row, index in enumerate(sample_indices):
            noisy, clean = dataset[index]
            noisy_batch = noisy.unsqueeze(0).to(device)
            denoised = denoise_tensor(model, noisy_batch)[0].cpu()
            psnr = calculate_psnr(clean, denoised)
            metrics_rows.append([str(index), f"{psnr:.2f} dB"])

            plt.subplot(sample_count, 3, row * 3 + 1)
            plt.imshow(noisy.permute(1, 2, 0).numpy())
            plt.title(f"Noisy block #{index}")
            plt.axis("off")

            plt.subplot(sample_count, 3, row * 3 + 2)
            plt.imshow(denoised.permute(1, 2, 0).numpy())
            plt.title("Denoised")
            plt.axis("off")

            plt.subplot(sample_count, 3, row * 3 + 3)
            plt.imshow(clean.permute(1, 2, 0).numpy())
            plt.title("GT")
            plt.axis("off")

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 1.2 * sample_count + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=metrics_rows,
        colLabels=["Validation Block", "PSNR"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title("SIDD Validation Metrics", pad=12)
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    model = UNetTransformer(
        base_channels=args.base_channels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        transformer_layers=args.transformer_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    if args.input is None:
        if args.validate_all:
            validate_all_blocks(
                model,
                device,
                noisy_mat=args.val_noisy_mat,
                gt_mat=args.val_gt_mat,
                batch_size=args.val_batch_size,
                metrics_output=args.metrics_output,
            )
            return

        show_validation_samples(
            model,
            device,
            noisy_mat=args.val_noisy_mat,
            gt_mat=args.val_gt_mat,
            num_samples=args.num_samples,
        )
        return

    image, noisy = load_image_tensor(args.input)
    noisy = noisy.to(device)
    with torch.no_grad():
        output = denoise_tensor(model, noisy)

    output_image = tensor_to_pil(output)
    if args.target is not None:
        _, clean = load_image_tensor(args.target)
        psnr = calculate_psnr(clean.squeeze(0), output.squeeze(0).cpu())
        print(f"PSNR: {psnr:.2f} dB")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_image.save(args.output)
        print(f"saved output to: {args.output}")
    else:
        columns = 2 if args.target is None else 3
        plt.figure(figsize=(5 * columns, 4))
        plt.subplot(1, columns, 1)
        plt.imshow(image)
        plt.title("Noisy")
        plt.axis("off")

        plt.subplot(1, columns, 2)
        plt.imshow(output_image)
        plt.title("Denoised")
        plt.axis("off")

        if args.target is not None:
            target_image = Image.open(args.target).convert("RGB")
            plt.subplot(1, columns, 3)
            plt.imshow(target_image)
            plt.title("GT")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ut_project.config import DEFAULT_DATA_ROOT, NOISE_PRESETS
from ut_project.engine.trainer import add_random_noise, calculate_psnr
from ut_project.models import UNetTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a UNet + Transformer denoising model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset root containing test/ when --input is not provided.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--noise-level",
        choices=["custom", *NOISE_PRESETS.keys()],
        default="custom",
        help="Noise preset for random test mode. 'custom' keeps --noise-min/--noise-max.",
    )
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def load_image_tensor(image_path: Path):
    image = Image.open(image_path).convert("RGB")
    x = transforms.ToTensor()(image).unsqueeze(0)
    return image, x


def show_random_test_samples(model, device, data_root: Path, noise_min: float, noise_max: float):
    test_dir = data_root / "test"
    image_paths = sorted(
        path for path in test_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    if not image_paths:
        raise ValueError(f"No images found in: {test_dir}")

    num_samples = min(3, len(image_paths))
    sample_paths = random.sample(image_paths, k=num_samples)

    plt.figure(figsize=(12, 4 * num_samples))
    metrics_rows = []

    with torch.no_grad():
        for row, image_path in enumerate(sample_paths):
            image, clean = load_image_tensor(image_path)
            clean = clean.to(device)
            noisy, _, noise_level = add_random_noise(clean, noise_min, noise_max)
            pred_noise = model(noisy)
            denoised = torch.clamp(noisy - pred_noise, 0.0, 1.0)

            clean_cpu = clean[0].cpu()
            noisy_cpu = noisy[0].cpu()
            denoised_cpu = denoised[0].cpu()
            psnr = calculate_psnr(clean_cpu, denoised_cpu)
            height, width = image.height, image.width
            metrics_rows.append(
                [
                    image_path.name,
                    f"{width}x{height}",
                    f"{noise_level[0].item():.3f}",
                    f"{psnr:.2f} dB",
                ]
            )

            plt.subplot(num_samples, 3, row * 3 + 1)
            plt.imshow(image)
            plt.title(f"Clean\n{image_path.name}")
            plt.axis("off")

            plt.subplot(num_samples, 3, row * 3 + 2)
            plt.imshow(noisy_cpu.permute(1, 2, 0).numpy())
            plt.title(f"Noisy\nstd={noise_level[0].item():.3f}")
            plt.axis("off")

            plt.subplot(num_samples, 3, row * 3 + 3)
            plt.imshow(denoised_cpu.permute(1, 2, 0).numpy())
            plt.title("Denoised")
            plt.axis("off")

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 1.2 * num_samples + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=metrics_rows,
        colLabels=["Image", "Size", "Noise Std", "PSNR"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title("Inference Metrics", pad=12)
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    device = torch.device("cuda")
    print(f"using device: {device}")
    if args.noise_level == "custom":
        noise_min = args.noise_min
        noise_max = args.noise_max
    else:
        noise_min, noise_max = NOISE_PRESETS[args.noise_level]

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
        print(f"test noise range: [{noise_min:.3f}, {noise_max:.3f}] (mode={args.noise_level})")
        show_random_test_samples(
            model,
            device,
            data_root=args.data_root,
            noise_min=noise_min,
            noise_max=noise_max,
        )
        return

    image, x = load_image_tensor(args.input)
    x = x.to(device)

    with torch.no_grad():
        pred_noise = model(x)
        output = torch.clamp(x - pred_noise, 0.0, 1.0)

    output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_image.save(args.output)
        print(f"saved output to: {args.output}")
    else:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Input")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(output_image)
        plt.title("Denoised")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ut_project.config import DEFAULT_BENCHMARK_BLOCKS, DEFAULT_BENCHMARK_ROOT
from ut_project.models import UNetTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Create SIDD benchmark SubmitSrgb.mat from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--blocks-mat", type=Path, default=DEFAULT_BENCHMARK_BLOCKS)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "image" / "Submit" / "SubmitSrgb.mat")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--method-name", type=str, default="UNetTransformer")
    parser.add_argument("--authors", type=str, default="")
    return parser.parse_args()


def require_scipy():
    try:
        from scipy.io import loadmat, savemat
    except ImportError as exc:
        raise ImportError("Creating SubmitSrgb.mat requires scipy. Install it with: pip install scipy") from exc
    return loadmat, savemat


def load_benchmark_blocks(blocks_mat: Path):
    loadmat, _ = require_scipy()
    data = loadmat(blocks_mat)
    if "BenchmarkBlocks32" not in data:
        raise ValueError(f"BenchmarkBlocks32 not found in: {blocks_mat}")
    blocks = np.asarray(data["BenchmarkBlocks32"], dtype=np.int64)
    if blocks.ndim != 2 or blocks.shape[1] != 4:
        raise ValueError(f"Expected BenchmarkBlocks32 to have shape Nx4, got: {blocks.shape}")
    return blocks


def load_model(args, device):
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
    return model


def denoise_batch(model, blocks, device):
    tensors = [transforms.ToTensor()(Image.fromarray(block)).unsqueeze(0) for block in blocks]
    noisy = torch.cat(tensors, dim=0).to(device)
    with torch.no_grad():
        pred_noise = model(noisy)
        denoised = torch.clamp(noisy - pred_noise, 0.0, 1.0)
    denoised = denoised.cpu().permute(0, 2, 3, 1).numpy()
    return np.clip(denoised * 255.0 + 0.5, 0, 255).astype(np.uint8)


def find_srgb_image(image_dir: Path):
    prefix = image_dir.name.split("_")[0]
    image_path = image_dir / f"{prefix}_NOISY_SRGB_010.PNG"
    if image_path.exists():
        return image_path

    matches = sorted(image_dir.glob("*_NOISY_SRGB_010.PNG"))
    if not matches:
        raise FileNotFoundError(f"No benchmark sRGB noisy image found in: {image_dir}")
    return matches[0]


def main():
    args = parse_args()
    _, savemat = require_scipy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    model = load_model(args, device)
    benchmark_dirs = sorted(path for path in args.benchmark_root.iterdir() if path.is_dir())[:40]
    if len(benchmark_dirs) != 40:
        raise ValueError(f"Expected 40 SIDD benchmark directories, found {len(benchmark_dirs)} in {args.benchmark_root}")

    block_specs = load_benchmark_blocks(args.blocks_mat)
    denoised_cells = np.empty((len(benchmark_dirs), len(block_specs)), dtype=object)
    elapsed = 0.0
    pixels = 0

    for image_index, image_dir in enumerate(benchmark_dirs):
        image_path = find_srgb_image(image_dir)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        pending_blocks = []
        pending_indices = []

        for block_index, (row, col, height, width) in enumerate(block_specs):
            top = row - 1
            left = col - 1
            block = image[top : top + height, left : left + width, :]
            pending_blocks.append(block)
            pending_indices.append(block_index)

            if len(pending_blocks) == args.batch_size or block_index == len(block_specs) - 1:
                t0 = time.perf_counter()
                denoised_blocks = denoise_batch(model, pending_blocks, device)
                elapsed += time.perf_counter() - t0

                for idx, denoised_block in zip(pending_indices, denoised_blocks):
                    denoised_cells[image_index, idx] = denoised_block
                    pixels += denoised_block.shape[0] * denoised_block.shape[1]

                pending_blocks = []
                pending_indices = []

        print(f"denoised benchmark image {image_index + 1:02d}/40: {image_path.name}")

    time_mp = elapsed * 1024 * 1024 / max(pixels, 1)
    optional_data = {
        "MethodName": args.method_name,
        "Authors": args.authors,
        "PaperTitle": "",
        "Venue": "SIDD Benchmark",
        "MachineSpecs": f"Python/PyTorch on {device}",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        args.output,
        {
            "DenoisedBlocksSrgb": denoised_cells,
            "TimeMPSrgb": np.array([[time_mp]], dtype=np.float64),
            "OptionalData": optional_data,
        },
        do_compression=True,
    )
    print(f"saved SIDD sRGB submission to: {args.output}")
    print(f"TimeMPSrgb: {time_mp:.6f} seconds/megapixel")


if __name__ == "__main__":
    main()

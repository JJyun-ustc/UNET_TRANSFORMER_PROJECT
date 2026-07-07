import argparse
import base64
import csv
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ut_project.models import UNetTransformer


BENCHMARK_URL = "http://130.63.97.225/share/SIDD_Blocks/BenchmarkNoisyBlocksSrgb.mat"


def parse_args():
    parser = argparse.ArgumentParser(description="Create Kaggle SIDD benchmark SubmitSrgb.csv from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, default=PROJECT_ROOT / "BenchmarkNoisyBlocksSrgb.mat")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "SubmitSrgb.csv")
    parser.add_argument("--url", type=str, default=BENCHMARK_URL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument("--limit-blocks", type=int, default=None, help="Optional smoke-test limit; omit for final submission.")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def require_scipy():
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("Reading BenchmarkNoisyBlocksSrgb.mat requires scipy. Install it with: pip install scipy") from exc
    return loadmat


def download_if_needed(input_file: Path, url: str):
    if input_file.exists():
        print(f"{input_file} exists. No need to download it.")
        return

    input_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading input file {input_file.name}...")
    urllib.request.urlretrieve(url, input_file)
    print("Downloaded successfully.")


def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    return base64_bytes.decode("utf-8")


def base64string_to_array(base64string, array_dtype, array_shape):
    decoded_bytes = base64.b64decode(base64string)
    decoded_array = np.frombuffer(decoded_bytes, dtype=array_dtype)
    return decoded_array.reshape(array_shape)


def load_inputs(input_file: Path):
    loadmat = require_scipy()
    key = "BenchmarkNoisyBlocksSrgb"
    data = loadmat(input_file)
    if key not in data:
        visible_keys = [name for name in data if not name.startswith("__")]
        raise ValueError(f"{key} not found in {input_file}. Available variables: {visible_keys}")

    inputs = data[key]
    if inputs.ndim != 5:
        raise ValueError(f"Expected {key} to have shape NxMxHxWxC, got {inputs.shape}")

    print(f"inputs.shape = {inputs.shape}")
    print(f"inputs.dtype = {inputs.dtype}")
    return inputs


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


def blocks_to_tensor(blocks):
    array = np.stack(blocks, axis=0)
    if np.issubdtype(array.dtype, np.integer):
        scale = float(np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / scale
    else:
        array = array.astype(np.float32)

    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def tensor_to_blocks(tensor, dtype):
    array = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    array = np.clip(array, 0.0, 1.0)

    if np.issubdtype(dtype, np.integer):
        scale = float(np.iinfo(dtype).max)
        return np.clip(array * scale + 0.5, 0, scale).astype(dtype)

    return array.astype(dtype)


def denoise_blocks(model, blocks, dtype, device):
    noisy = blocks_to_tensor(blocks).to(device)
    with torch.no_grad():
        pred_noise = model(noisy)
        denoised = torch.clamp(noisy - pred_noise, 0.0, 1.0)
    return tensor_to_blocks(denoised, dtype)


def write_submission_csv(model, inputs, output_file: Path, batch_size: int, device, progress_every: int, limit_blocks: int | None):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    total_blocks = inputs.shape[0] * inputs.shape[1]
    if limit_blocks is not None:
        total_blocks = min(total_blocks, limit_blocks)

    print(f"Saving outputs to {output_file}")
    print(f"Number of blocks = {total_blocks}")

    pending_blocks = []
    pending_ids = []
    written = 0
    start_time = time.perf_counter()

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "BLOCK"])

        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                if limit_blocks is not None and written + len(pending_blocks) >= limit_blocks:
                    break

                block_id = i * inputs.shape[1] + j
                in_block = inputs[i, j, :, :, :]
                pending_blocks.append(in_block)
                pending_ids.append(block_id)

                if len(pending_blocks) == batch_size:
                    written = flush_batch(model, writer, pending_blocks, pending_ids, device, written, progress_every, total_blocks)
                    pending_blocks = []
                    pending_ids = []

            if limit_blocks is not None and written + len(pending_blocks) >= limit_blocks:
                break

        if pending_blocks:
            written = flush_batch(model, writer, pending_blocks, pending_ids, device, written, progress_every, total_blocks)

    elapsed = time.perf_counter() - start_time
    print(f"Wrote {written} blocks in {elapsed:.2f} seconds.")


def flush_batch(model, writer, pending_blocks, pending_ids, device, written, progress_every, total_blocks):
    dtype = pending_blocks[0].dtype
    denoised_blocks = denoise_blocks(model, pending_blocks, dtype=dtype, device=device)

    for block_id, in_block, out_block in zip(pending_ids, pending_blocks, denoised_blocks):
        assert in_block.shape == out_block.shape
        assert in_block.dtype == out_block.dtype
        writer.writerow([block_id, array_to_base64string(out_block)])
        written += 1

    if progress_every > 0 and (written % progress_every == 0 or written == total_blocks):
        print(f"processed {written}/{total_blocks} blocks")

    return written


def main():
    args = parse_args()
    download_if_needed(args.input_file, args.url)
    inputs = load_inputs(args.input_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    model = load_model(args, device)

    write_submission_csv(
        model=model,
        inputs=inputs,
        output_file=args.output,
        batch_size=args.batch_size,
        device=device,
        progress_every=args.progress_every,
        limit_blocks=args.limit_blocks,
    )
    print("Submit the output file SubmitSrgb.csv at")
    print("kaggle.com/competitions/sidd-benchmark-srgb-psnr")
    print("Done.")


if __name__ == "__main__":
    main()

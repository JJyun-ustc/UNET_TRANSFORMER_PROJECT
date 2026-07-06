import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _to_rgb_tensor(image: Image.Image):
    return F.to_tensor(image.convert("RGB"))


def _read_sidd_mat_array(mat_path: Path, preferred_names: tuple[str, ...]):
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("Reading SIDD .mat files requires scipy. Install it with: pip install scipy") from exc

    mat_path = Path(mat_path)
    data = loadmat(mat_path)
    for name in preferred_names:
        if name in data:
            return data[name]

    visible_keys = [key for key in data if not key.startswith("__")]
    if not visible_keys:
        raise ValueError(f"No array variables found in: {mat_path}")
    return data[visible_keys[0]]


def _iter_mat_blocks(array):
    if array.dtype == object:
        for item in array.reshape(-1):
            yield np.asarray(item)
        return

    array = np.asarray(array)
    if array.ndim in {4, 5}:
        channel_axis = next((axis for axis, size in enumerate(array.shape) if size in {1, 3}), None)
        if channel_axis is None:
            raise ValueError(f"Cannot infer channel axis from validation array shape: {array.shape}")
        array = np.moveaxis(array, channel_axis, -1)

        candidate_axes = list(range(array.ndim - 1))
        spatial_axes = sorted(candidate_axes, key=lambda axis: array.shape[axis], reverse=True)[:2]
        sample_axes = [axis for axis in candidate_axes if axis not in spatial_axes]
        array = np.transpose(array, sample_axes + spatial_axes + [array.ndim - 1])

        for item in array.reshape(-1, *array.shape[-3:]):
            yield item
        return

    raise ValueError(f"Unsupported SIDD validation array shape: {array.shape}")


def _mat_block_to_image(block) -> Image.Image:
    block = np.asarray(block)
    if block.ndim == 2:
        block = np.repeat(block[..., None], repeats=3, axis=-1)
    if block.ndim != 3:
        raise ValueError(f"Expected an HxWxC image block, got shape: {block.shape}")

    if block.shape[0] in {1, 3} and block.shape[-1] not in {1, 3}:
        block = np.moveaxis(block, 0, -1)
    if block.shape[-1] == 1:
        block = np.repeat(block, repeats=3, axis=-1)
    if block.shape[-1] != 3:
        raise ValueError(f"Expected an RGB block, got shape: {block.shape}")

    if np.issubdtype(block.dtype, np.floating):
        block = np.clip(block, 0.0, 1.0) * 255.0
    block = np.clip(block, 0, 255).astype(np.uint8)
    return Image.fromarray(block, mode="RGB")


class SIDDMediumDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        patch_size: int | None = None,
        augment: bool = True,
        pairs_per_image: int = 1,
    ):
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.pairs_per_image = max(1, pairs_per_image)
        self.pairs = self._find_pairs()
        if not self.pairs:
            raise ValueError(f"No SIDD noisy/GT image pairs found in: {self.root_dir}")

    def _find_pairs(self):
        noisy_paths = sorted(
            path for path in self.root_dir.rglob("*_NOISY_SRGB_*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        pairs = []
        for noisy_path in noisy_paths:
            gt_path = noisy_path.with_name(noisy_path.name.replace("_NOISY_SRGB_", "_GT_SRGB_"))
            if gt_path.exists():
                pairs.append((noisy_path, gt_path))
        return pairs

    def __len__(self):
        return len(self.pairs) * self.pairs_per_image

    def __getitem__(self, index):
        noisy_path, gt_path = self.pairs[index % len(self.pairs)]
        noisy = Image.open(noisy_path).convert("RGB")
        clean = Image.open(gt_path).convert("RGB")

        if noisy.size != clean.size:
            raise ValueError(f"Image pair size mismatch: {noisy_path} vs {gt_path}")

        if self.patch_size is not None:
            noisy, clean = self._paired_random_crop(noisy, clean, self.patch_size)

        if self.augment:
            if random.random() < 0.5:
                noisy = F.hflip(noisy)
                clean = F.hflip(clean)
            if random.random() < 0.5:
                noisy = F.vflip(noisy)
                clean = F.vflip(clean)

        return _to_rgb_tensor(noisy), _to_rgb_tensor(clean)

    @staticmethod
    def _paired_random_crop(noisy: Image.Image, clean: Image.Image, patch_size: int):
        width, height = noisy.size
        if width < patch_size or height < patch_size:
            pad_width = max(patch_size - width, 0)
            pad_height = max(patch_size - height, 0)
            padding = (0, 0, pad_width, pad_height)
            noisy = F.pad(noisy, padding, padding_mode="reflect")
            clean = F.pad(clean, padding, padding_mode="reflect")
            width, height = noisy.size

        left = random.randint(0, width - patch_size)
        top = random.randint(0, height - patch_size)
        noisy = F.crop(noisy, top, left, patch_size, patch_size)
        clean = F.crop(clean, top, left, patch_size, patch_size)
        return noisy, clean


class SIDDValidationBlocksDataset(Dataset):
    def __init__(self, noisy_mat: Path, gt_mat: Path):
        noisy_array = _read_sidd_mat_array(
            noisy_mat,
            preferred_names=("ValidationNoisyBlocksSrgb", "ValidationNoisyBlocks"),
        )
        gt_array = _read_sidd_mat_array(
            gt_mat,
            preferred_names=("ValidationGtBlocksSrgb", "ValidationGtBlocks"),
        )
        self.noisy_blocks = list(_iter_mat_blocks(noisy_array))
        self.clean_blocks = list(_iter_mat_blocks(gt_array))
        if len(self.noisy_blocks) != len(self.clean_blocks):
            raise ValueError(
                f"Validation block count mismatch: {len(self.noisy_blocks)} noisy vs "
                f"{len(self.clean_blocks)} GT"
            )
        if not self.noisy_blocks:
            raise ValueError(f"No validation blocks loaded from: {noisy_mat}")

    def __len__(self):
        return len(self.noisy_blocks)

    def __getitem__(self, index):
        noisy = _mat_block_to_image(self.noisy_blocks[index])
        clean = _mat_block_to_image(self.clean_blocks[index])
        return _to_rgb_tensor(noisy), _to_rgb_tensor(clean)


def build_sidd_train_dataloader(
    root_dir: Path,
    batch_size: int,
    patch_size: int | None,
    shuffle: bool,
    num_workers: int,
    pairs_per_image: int = 1,
):
    dataset = SIDDMediumDataset(
        root_dir=root_dir,
        patch_size=patch_size,
        augment=shuffle,
        pairs_per_image=pairs_per_image,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_sidd_validation_dataloader(noisy_mat: Path, gt_mat: Path, batch_size: int, num_workers: int):
    dataset = SIDDValidationBlocksDataset(noisy_mat=noisy_mat, gt_mat=gt_mat)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

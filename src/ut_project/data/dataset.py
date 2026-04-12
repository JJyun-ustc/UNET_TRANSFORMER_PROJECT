from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: Path, patch_size: int | None = None):
        self.root_dir = Path(root_dir)
        self.image_paths = sorted(
            path for path in self.root_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No images found in: {self.root_dir}")

        transform_ops = []
        if patch_size is not None:
            transform_ops.append(transforms.RandomCrop(patch_size, pad_if_needed=True))
        transform_ops.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_ops)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


def build_dataloader(root_dir: Path, batch_size: int, patch_size: int | None, shuffle: bool, num_workers: int):
    dataset = ImageFolderDataset(root_dir=root_dir, patch_size=patch_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

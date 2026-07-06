import random
from pathlib import Path

import torch
from torch import nn


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor):
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(torch.tensor(1.0 / mse)).item()


class Trainer:
    def __init__(self, model: nn.Module, config, train_loader, test_loader, device: torch.device):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_psnr(self):
        self.model.eval()
        total_psnr = 0.0
        total_images = 0

        with torch.no_grad():
            for noisy, clean in self.test_loader:
                noisy = noisy.to(self.device, non_blocking=True)
                clean = clean.to(self.device, non_blocking=True)
                pred_noise = self.model(noisy)
                denoised = torch.clamp(noisy - pred_noise, 0.0, 1.0)

                for clean_image, denoised_image in zip(clean, denoised):
                    total_psnr += calculate_psnr(clean_image, denoised_image)
                    total_images += 1

        return total_psnr / max(total_images, 1)

    def train(self):
        best_psnr = float("-inf")

        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0

            for step, (noisy, clean) in enumerate(self.train_loader, start=1):
                noisy = noisy.to(self.device, non_blocking=True)
                clean = clean.to(self.device, non_blocking=True)
                target_noise = noisy - clean
                pred_noise = self.model(noisy)
                loss = self.criterion(pred_noise, target_noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                if step % 20 == 0:
                    print(
                        f"epoch {epoch + 1}/{self.config.epochs}, "
                        f"step {step}/{len(self.train_loader)}, "
                        f"loss = {loss.item():.6f}"
                    )

            epoch_loss /= max(len(self.train_loader), 1)
            avg_psnr = self.evaluate_psnr()
            print(
                f"epoch {epoch + 1}/{self.config.epochs}, "
                f"avg_loss = {epoch_loss:.6f}, "
                f"avg_psnr = {avg_psnr:.2f} dB"
            )

            latest_path = self.checkpoint_dir / "unet_transformer_latest.pth"
            torch.save(self.model.state_dict(), latest_path)

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_path = self.checkpoint_dir / "unet_transformer_best.pth"
                torch.save(self.model.state_dict(), best_path)
                print(f"best model saved to: {best_path} (val_psnr = {best_psnr:.2f} dB)")

import os
import math
import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

try:
    from _bootstrap import ensure_modeling_root_on_path
except ImportError:
    import sys

    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from _bootstrap import ensure_modeling_root_on_path

MODEL_ROOT = ensure_modeling_root_on_path()
from data.data_loader import load_data_sample


CLASS_NAMES = [
    "plastic pencil case 1",
    "plastic pencil case 2",
    "plastic placemat 1",
    "plastic placemat 2",
    "plastic placemat 3",
    "plastic placemat 4",
    "cork mat",
    "denim fabric 1",
    "denim fabric 2",
    "felt material",
    "polyester cloth",
    "cotton scarf",
]


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, reduced, kernel_size=1)
        self.fc2 = nn.Conv1d(reduced, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class MultiScaleStem1d(nn.Module):
    def __init__(self, in_channels: int, branch_channels: int = 32):
        super().__init__()
        kernels = (3, 7, 15)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        branch_channels,
                        kernel_size=k,
                        stride=1,
                        padding=(k - 1) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(branch_channels),
                    nn.SiLU(),
                )
                for k in kernels
            ]
        )
        fused_channels = branch_channels * len(kernels)
        self.fuse = nn.Sequential(
            nn.Conv1d(fused_channels, fused_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(fused_channels),
            nn.SiLU(),
        )
        self.out_channels = fused_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([b(x) for b in self.branches], dim=1)
        return self.fuse(x)


class ResidualDSBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.dw = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm1d(in_channels)

        self.pw = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm1d(out_channels)

        self.se = SqueezeExcite1d(out_channels)
        self.act = nn.SiLU()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        out = self.dw(x)
        out = self.dw_bn(out)
        out = self.act(out)

        out = self.pw(out)
        out = self.pw_bn(out)
        out = self.se(out)

        out = out + residual
        out = self.act(out)
        return out


class TactileCNNWithEmbedding(nn.Module):
    def __init__(self, in_channels: int = 13, num_classes: int = 12, embedding_dim: int = 96):
        super().__init__()
        self.stem = MultiScaleStem1d(in_channels=in_channels, branch_channels=32)

        c0 = self.stem.out_channels
        self.blocks = nn.Sequential(
            ResidualDSBlock1d(c0, 128, kernel_size=7, stride=1),
            ResidualDSBlock1d(128, 192, kernel_size=5, stride=2),
            ResidualDSBlock1d(192, 256, kernel_size=5, stride=2),
            ResidualDSBlock1d(256, 320, kernel_size=3, stride=2),
            ResidualDSBlock1d(320, 384, kernel_size=3, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pre_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(384, 256),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(256, num_classes)
        self.embedding_head = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Expected input shape: (B, T, C) from your existing loader.
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.pre_head(x)

        logits = self.classifier(x)
        embedding = F.normalize(self.embedding_head(x), dim=1)
        return logits, embedding


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.15,
) -> torch.Tensor:
    labels = labels.view(-1, 1)
    batch_size = labels.size(0)

    if batch_size < 2:
        return embeddings.new_tensor(0.0)

    sim = torch.matmul(embeddings, embeddings.T) / temperature

    logits_mask = torch.ones_like(sim) - torch.eye(batch_size, device=sim.device)
    sim = sim - (1.0 - logits_mask) * 1e9

    positive_mask = (labels == labels.T).float() * logits_mask

    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0
    if valid_anchor.sum() == 0:
        return embeddings.new_tensor(0.0)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_count + 1e-12)

    loss = -mean_log_prob_pos[valid_anchor].mean()
    return loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits, _ = model(inputs)
        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def save_confusion_matrix(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_file: str,
) -> None:
    model.eval()
    all_preds = []
    all_labels = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits, _ = model(inputs)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)


def train(args: argparse.Namespace) -> Dict[str, list]:
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader = load_data_sample(
        dataset_path=args.dataset_path,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        smooth=args.smooth,
        window_size=args.smooth_window,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = TactileCNNWithEmbedding(
        in_channels=13,
        num_classes=12,
        embedding_dim=args.embedding_dim,
    ).to(device)

    print(f"Using device: {device}")
    print(f"Trainable parameters: {count_trainable_params(model):,}")

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "train_ce": [],
        "train_supcon": [],
    }

    best_val_acc = -math.inf
    best_path = os.path.join(args.output_dir, "best_multiscale_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        running_ce = 0.0
        running_supcon = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits, embeddings = model(inputs)

            ce = ce_criterion(logits, targets)
            supcon = supervised_contrastive_loss(
                embeddings,
                targets,
                temperature=args.supcon_temperature,
            )
            loss = ce + args.supcon_weight * supcon

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_ce += ce.item() * batch_size
            running_supcon += supcon.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_samples += batch_size

        scheduler.step()

        train_loss = running_loss / total_samples
        train_ce = running_ce / total_samples
        train_supcon = running_supcon / total_samples
        train_acc = total_correct / total_samples

        val_loss, val_acc = evaluate(model, val_loader, device, ce_criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["train_ce"].append(train_ce)
        history["train_supcon"].append(train_supcon)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} (CE {train_ce:.4f} + SupCon {train_supcon:.4f}) | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    final_path = os.path.join(args.output_dir, "last_multiscale_model.pt")
    torch.save(model.state_dict(), final_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    save_confusion_matrix(
        model,
        val_loader,
        device,
        os.path.join(args.output_dir, "best_confusion_matrix.png"),
    )

    x = range(1, args.epochs + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(x, history["train_loss"], label="Train Loss")
    axs[0].plot(x, history["val_loss"], label="Val Loss")
    axs[0].set_title("Loss")
    axs[0].legend()

    axs[1].plot(x, history["train_acc"], label="Train Acc")
    axs[1].plot(x, history["val_acc"], label="Val Acc")
    axs[1].set_title("Accuracy")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_summary.png"))
    plt.close(fig)

    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Saved best model to: {best_path}")
    print(f"Saved final model to: {final_path}")

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-scale residual DS-CNN with CE + SupCon")

    parser.add_argument("--dataset-path", type=str, default=str(MODEL_ROOT / "data" / "int8_dataset.npz"))
    parser.add_argument("--output-dir", type=str, default=str(MODEL_ROOT / "gallery_assets" / "multiscale_supcon"))

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--supcon-weight", type=float, default=0.10)
    parser.add_argument("--supcon-temperature", type=float, default=0.15)
    parser.add_argument("--embedding-dim", type=int, default=96)

    parser.add_argument("--sample-size", type=float, default=1.0)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=13)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

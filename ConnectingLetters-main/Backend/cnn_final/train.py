import os

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import train_loader, val_loader
from model import build_model


CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
PHASE1_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_phase1.pt")
PHASE2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_phase2.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = nn.CrossEntropyLoss(label_smoothing=0.1)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc="Train", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            loss = CRITERION(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(loader, desc="Val", leave=False)

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = CRITERION(outputs, labels)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    val_loss = running_loss / len(loader.dataset)
    correct = sum(int(pred == label) for pred, label in zip(all_preds, all_labels))
    val_acc = correct / len(all_labels)
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    return val_loss, val_acc, val_f1


def save_checkpoint(model, optimizer, epoch, best_val_f1, val_acc, checkpoint_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": best_val_f1,
            "val_acc": val_acc,
        },
        checkpoint_path,
    )


def run_phase(
    model,
    optimizer,
    scheduler,
    scaler,
    writer,
    start_epoch,
    num_epochs,
    checkpoint_path,
    initial_best_val_f1=float("-inf"),
):
    best_val_f1 = initial_best_val_f1

    for epoch_idx in range(num_epochs):
        global_epoch = start_epoch + epoch_idx
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, DEVICE)
        val_loss, val_acc, val_f1 = validate(model, val_loader, DEVICE)

        writer.add_scalar("Loss/train", train_loss, global_epoch)
        writer.add_scalar("Loss/val", val_loss, global_epoch)
        writer.add_scalar("Accuracy/val", val_acc, global_epoch)
        writer.add_scalar("F1_macro/val", val_f1, global_epoch)

        saved = False
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=global_epoch,
                best_val_f1=best_val_f1,
                val_acc=val_acc,
                checkpoint_path=checkpoint_path,
            )
            saved = True

        saved_suffix = " | ⭐ Saved" if saved else ""
        print(
            f"Epoch {global_epoch}/{start_epoch + num_epochs - 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val F1: {val_f1:.4f}"
            f"{saved_suffix}"
        )

    return best_val_f1


def train_phase1(model, writer, scaler):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

    return run_phase(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        writer=writer,
        start_epoch=1,
        num_epochs=15,
        checkpoint_path=PHASE1_CHECKPOINT,
    )


def train_phase2(model, writer, scaler):
    checkpoint = torch.load(PHASE1_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    for param in model.parameters():
        param.requires_grad = False

    for param in model.blocks[4].parameters():
        param.requires_grad = True
    for param in model.blocks[5].parameters():
        param.requires_grad = True
    for param in model.blocks[6].parameters():
        param.requires_grad = True
    for param in model.conv_head.parameters():
        param.requires_grad = True
    for param in model.bn2.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

    return run_phase(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        writer=writer,
        start_epoch=16,
        num_epochs=25,
        checkpoint_path=PHASE2_CHECKPOINT,
    )


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    model = build_model(DEVICE)
    writer = SummaryWriter(LOG_DIR)
    scaler = GradScaler(enabled=DEVICE.type == "cuda")

    train_phase1(model, writer, scaler)
    train_phase2(model, writer, scaler)

    writer.close()
    print("Training complete. Best model saved to checkpoints/best_phase2.pt")


if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    model = build_model(DEVICE)
    writer = SummaryWriter(LOG_DIR)
    scaler = GradScaler(enabled=DEVICE.type == "cuda")

    if os.path.exists("checkpoints/best_phase1.pt"):
        print("Phase 1 checkpoint found. Skipping to Phase 2...")
        train_phase2(model, writer, scaler)
    else:
        print("No Phase 1 checkpoint found. Starting from Phase 1...")
        train_phase1(model, writer, scaler)
        train_phase2(model, writer, scaler)

    writer.close()
    print("Training complete. Best model saved to checkpoints/best_phase2.pt")

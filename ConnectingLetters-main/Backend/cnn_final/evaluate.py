import torch
from torch import nn

from config import load_config
from dataset import build_dataloaders
from model import EmotionCNN
from src.metrics import build_classification_report


def main():
    config = load_config()
    _, _, test_loader = build_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionCNN(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = sum(int(t == p) for t, p in zip(y_true, y_pred)) / len(y_true)

    print(f"test_loss={avg_loss:.4f}")
    print(f"test_acc={accuracy:.4f}")
    print(build_classification_report(y_true, y_pred, config["class_names"]))


if __name__ == "__main__":
    main()

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class_to_idx = {
    "CONFUSED": 0,
    "FRUSTRATED": 1,
    "HAPPY": 2,
    "NEUTRAL": 3,
}

_IDX_TO_CLASS = {index: name for name, index in class_to_idx.items()}
_DATA_ROOT = Path(__file__).resolve().parent.parent / "game_data"
_TRAIN_DIR = _DATA_ROOT / "train"
_VAL_DIR = _DATA_ROOT / "validation"
_TEST_DIR = _DATA_ROOT / "test"
_IMAGE_SIZE = (224, 224)
_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]


class FixedClassImageFolder(datasets.ImageFolder):
    """ImageFolder that enforces a fixed class-to-index mapping."""

    def find_classes(self, directory):
        available_classes = {
            entry.name for entry in Path(directory).iterdir() if entry.is_dir()
        }
        expected_classes = set(class_to_idx)

        missing_classes = expected_classes - available_classes
        extra_classes = available_classes - expected_classes
        if missing_classes or extra_classes:
            details = []
            if missing_classes:
                details.append(f"missing={sorted(missing_classes)}")
            if extra_classes:
                details.append(f"extra={sorted(extra_classes)}")
            raise FileNotFoundError(
                f"Class folders in '{directory}' do not match the required mapping: "
                + ", ".join(details)
            )

        classes = [_IDX_TO_CLASS[index] for index in range(len(class_to_idx))]
        return classes, class_to_idx.copy()


def build_train_transform():
    return transforms.Compose(
        [
            transforms.Resize(_IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )


def build_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=_NORM_MEAN, std=_NORM_STD),
        ]
    )


def build_datasets():
    train_dataset = FixedClassImageFolder(
        root=_TRAIN_DIR,
        transform=build_train_transform(),
    )
    val_dataset = FixedClassImageFolder(
        root=_VAL_DIR,
        transform=build_eval_transform(),
    )
    test_dataset = FixedClassImageFolder(
        root=_TEST_DIR,
        transform=build_eval_transform(),
    )
    return train_dataset, val_dataset, test_dataset


def build_dataloaders():
    train_dataset, val_dataset, test_dataset = build_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = build_dataloaders()


if __name__ == "__main__":
    images, labels = next(iter(train_loader))
    print(f"image tensor shape: {images.shape}")
    print(f"labels tensor: {labels}")
    print(f"class_to_idx: {class_to_idx}")

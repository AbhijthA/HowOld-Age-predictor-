"""
Train an age-classification model on FairFace using a ResNet backbone (PyTorch).

Usage example:
    python fairface_age_resnet.py \
      --data-root /path/to/FairFace \
      --train-csv annotations/fairface_label_train.csv \
      --val-csv annotations/fairface_label_val.csv \
      --epochs 12 --batch-size 64 --lr 1e-3 --model-out output/resnet_age_best.pth

Notes:
- Uses images from fairface-img-margin125-trainval/{train,val}/ directories.
- Expects CSVs with columns: file_name, age, gender, race, etc. (we use file_name and age)
- Age labels are mapped to 9 classes: ['0-2','3-9','10-19','20-29','30-39','40-49','50-59','60-69','70+']
- Requires torchvision, torch, tqdm, Pillow
- Will use CUDA if available.

Author: ChatGPT (example script, adapt paths/augmentations/hyperparams to your setup)
"""

import os
import argparse
import time
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from tqdm import tqdm


# Original age classes (excluding 0;2 and 3;9 for training)
ALL_AGE_CLASSES = [
    "0;2", "3;9", "10;19", "20;29",
    "30;39", "40;49", "50;59", "60;69", "70;120"
]

# Training age classes (excluding first two)
AGE_CLASSES = [
    "10;19", "20;29", "30;39", "40;49", 
    "50;59", "60;69", "70;120"
]

AGE_TO_IDX = {a: i for i, a in enumerate(AGE_CLASSES)}
EXCLUDED_AGES = {"0;2", "3;9"}


class FairFaceAgeDataset(Dataset):
    """Simple dataset for FairFace age classification.

    Expects a CSV file with at least a column 'file_name' and a column 'age' where 'age'
    already uses the bucket labels from AGE_CLASSES. If your CSV stores numeric ages,
    you'll need to bin them into the correct class before using this script.
    """

    def __init__(self, data_root, csv_path, images_dir_name, transform=None):
        import csv
        self.data_root = data_root
        self.csv_path = csv_path
        self.images_dir = os.path.join(data_root, images_dir_name)
        self.transform = transform

        # read csv
        self.samples = []  # list of (image_path, label_idx)
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                fname = r.get('img_name') or r.get('file_name') or r.get('file') or r.get('image')
                age_label = r.get('age')
                if fname is None or age_label is None:
                    # skip bad rows
                    continue
                # Trim possible whitespace
                age_label = age_label.strip()
                
                # Skip excluded age groups
                if age_label in EXCLUDED_AGES:
                    continue
                    
                # If age_label is numeric, try to bin it
                if age_label in AGE_TO_IDX:
                    idx = AGE_TO_IDX[age_label]
                else:
                    # try numeric conversion and bin
                    try:
                        ag = float(age_label)
                        idx = self._bin_age_numeric(ag)
                        # Skip if binned age falls in excluded categories
                        if idx == -1:
                            continue
                    except Exception:
                        # unknown label - skip
                        continue
                img_path = os.path.join(self.images_dir, fname)
                if not os.path.isfile(img_path):
                    # try with .jpg/.png variations
                    if os.path.isfile(img_path + '.jpg'):
                        img_path = img_path + '.jpg'
                    elif os.path.isfile(img_path + '.png'):
                        img_path = img_path + '.png'
                    else:
                        # missing file, skip
                        continue
                self.samples.append((img_path, idx))

        if len(self.samples) == 0:
            raise RuntimeError(f'No samples found. Check data_root (={data_root}), images_dir_name (={images_dir_name}) and csv_path (={csv_path}).')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    @staticmethod
    def _bin_age_numeric(age):
        # bins that match AGE_CLASSES (excluding 0;2 and 3;9)
        if age <= 2:
            return -1  # excluded
        if age <= 9:
            return -1  # excluded
        if age <= 19:
            return 0  # 10;19
        if age <= 29:
            return 1  # 20;29
        if age <= 39:
            return 2  # 30;39
        if age <= 49:
            return 3  # 40;49
        if age <= 59:
            return 4  # 50;59
        if age <= 69:
            return 5  # 60;69
        return 6  # 70;120


def make_transforms(train=True, image_size=224):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def compute_class_weights(dataset):
    counts = Counter()
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts.values())
    # inverse frequency
    weights = [0.0] * len(AGE_CLASSES)
    for i in range(len(AGE_CLASSES)):
        if counts[i] == 0:
            weights[i] = 0.0
        else:
            weights[i] = total / (len(AGE_CLASSES) * counts[i])
    return torch.tensor(weights, dtype=torch.float)


def build_model(num_classes=9, backbone='resnet18', pretrained=True):
    if backbone == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif backbone == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError('Unsupported backbone')
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    confmat = torch.zeros(len(AGE_CLASSES), len(AGE_CLASSES), dtype=torch.int64)
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confmat[t.long(), p.long()] += 1
    acc = correct / total if total > 0 else 0.0
    return acc, confmat


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for images, labels in tqdm(dataloader, desc='Train', leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    avg_loss = running_loss / total
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def save_checkpoint(state, is_best, filename='checkpoint.pth', bestname='best.pth'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    if is_best:
        torch.save(state, bestname)


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet on FairFace for age prediction')
    parser.add_argument('--data-root', type=str, required=True, help='Path to FairFace dir')
    parser.add_argument('--train-csv', type=str, required=True, help='Path to train csv')
    parser.add_argument('--val-csv', type=str, required=True, help='Path to val csv')
    parser.add_argument('--train-images-dir', type=str, default='fairface-img-margin025-trainval\train', help='Train images subdir inside data-root')
    parser.add_argument('--val-images-dir', type=str, default='fairface-img-margin025-trainval\val', help='Val images subdir inside data-root')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18','resnet50'])
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out', type=str, default='output/resnet_age_best.pth')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA even if available')
    return parser.parse_args()


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Using device:', device)

    train_tf = make_transforms(train=True, image_size=args.image_size)
    val_tf = make_transforms(train=False, image_size=args.image_size)

    train_dataset = FairFaceAgeDataset(args.data_root, args.train_csv, args.train_images_dir, transform=train_tf)
    val_dataset = FairFaceAgeDataset(args.data_root, args.val_csv, args.val_images_dir, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_cuda)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=use_cuda)

    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    model = build_model(num_classes=len(AGE_CLASSES), backbone=args.backbone, pretrained=True)
    model = model.to(device)

    # compute class weights from training set to help with imbalance
    class_weights = compute_class_weights(train_dataset).to(device)
    print('Class weights:', class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    scaler = torch.cuda.amp.GradScaler() if use_cuda else None

    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt.get('optim_state', optimizer.state_dict()))
        start_epoch = ckpt.get('epoch', 0)
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f'Resumed from {args.resume}, start_epoch={start_epoch}, best_val_acc={best_val_acc}')

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_acc, confmat = validate(model, val_loader, device)
        scheduler.step()

        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)

        print(f'Epoch {epoch+1}/{args.epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Val acc: {val_acc:.4f} | Best val: {best_val_acc:.4f} | Time: {time.time()-t0:.1f}s')

        # print simple per-class accuracy
        per_class_acc = confmat.diag().float() / confmat.sum(1).clamp(min=1)
        for i, acc in enumerate(per_class_acc.tolist()):
            print(f'  Class {i} ({AGE_CLASSES[i]}): {acc:.3f}  samples={int(confmat.sum(1)[i].item())}')

        # save checkpoint
        state = {
            'epoch': epoch+1,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
        }
        ckpt_name = os.path.join(os.path.dirname(args.out), f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint(state, is_best, filename=ckpt_name, bestname=args.out)

    print('Training finished. Best validation accuracy:', best_val_acc)


if __name__ == '__main__':
    main()

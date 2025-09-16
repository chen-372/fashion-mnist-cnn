import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from src.models.cnn import FashionCNN

CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def plot_confusion(cm: np.ndarray, out_path: Path):
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, range(len(CLASS_NAMES)), rotation=45, ha='right')
    plt.yticks(ticks, range(len(CLASS_NAMES)))
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--data-root', type=str, default='./data_cache')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FashionCNN().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    test_ds = datasets.FashionMNIST(root=args.data_root, train=False, transform=tfms, download=True)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    # Reports
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report)
    cm = confusion_matrix(y_true, y_pred)
    out_dir = Path('outputs'); out_dir.mkdir(exist_ok=True, parents=True)
    plot_confusion(cm, out_dir / 'confusion_matrix.png')

    # Sample predictions grid
    idxs = np.random.choice(len(test_ds), size=36, replace=False)
    imgs = torch.stack([test_ds[i][0] for i in idxs], dim=0)
    labels = [CLASS_NAMES[int(test_ds[i][1])] for i in idxs]
    model.eval()
    with torch.no_grad():
        preds = model(imgs.to(device)).argmax(dim=1).cpu().numpy()
    titles = [f"p:{CLASS_NAMES[int(p)]}\n t:{t}" for p, t in zip(preds, labels)]
    grid = utils.make_grid(imgs, nrow=6, padding=2)
    fig = plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1,2,0).numpy().squeeze())
    plt.axis('off')
    plt.title('Sample predictions (p=pred, t=true)')
    fig.savefig(out_dir / 'sample_predictions.png', bbox_inches='tight', dpi=150)
    plt.close(fig)

if __name__ == '__main__':
    main()

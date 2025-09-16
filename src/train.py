import argparse, json, time, random
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from torch.cuda import amp
from tqdm import tqdm
from rich import print as rprint

from src.data.datasets import build_dataloaders
from src.models.cnn import FashionCNN

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        bs = y.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(logits, y) * bs
        n += bs
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--momentum', type=float, default=0.9)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data-root', type=str, default='./data_cache')
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--amp', action='store_true', help='use mixed precision')
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rprint(f"[bold]Device:[/bold] {device}")

    out_dir = Path('outputs'); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'last_run_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed
    )

    model = FashionCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler(enabled=args.amp)

    best_val_acc, best_state, bad_epochs = 0.0, None, 0
    metrics_rows = ["epoch,train_loss,train_acc,val_loss,val_acc"]
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, args.amp)
        vl, va = evaluate(model, val_loader, criterion, device)
        rprint(f"[cyan]Epoch {epoch:02d}[/cyan] | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")
        metrics_rows.append(f"{epoch},{tl:.6f},{ta:.6f},{vl:.6f},{va:.6f}")

        if va > best_val_acc:
            best_val_acc = va
            best_state = model.state_dict()
            torch.save(best_state, out_dir / 'model.pt')
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                rprint(f"[yellow]Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs).[/yellow]")
                break

    # Save metrics csv
    (out_dir / 'metrics_train.csv').write_text("\n".join(metrics_rows))

    # Final test with best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    tl, ta = evaluate(model, train_loader, criterion, device)
    vl, va = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    rprint(f"[bold green]Test accuracy: {test_acc:.4f}[/bold green]")
    rprint(f"Artifacts saved in: {out_dir.resolve()}")

if __name__ == '__main__':
    main()

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from graph_hand import Graph
from dataset import KeypointDataset
from model import STGCN


def main():
    ap = argparse.ArgumentParser()

    # Paths
    ap.add_argument("--labels_csv", type=str, default="dataset/processed/labels_clean.csv")

    # Training config (your chosen defaults)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # Model config
    ap.add_argument("--use_xyz", action="store_true", help="Use x,y,z (default uses x,y only)")
    ap.add_argument("--num_class", type=int, default=35)

    # Output
    ap.add_argument("--save_dir", type=str, default="checkpoints")

    args = ap.parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Graph adjacency (K, V, V)
    graph = Graph(strategy="spatial", max_hop=1)
    A = graph.A

    # Model
    in_channels = 3 if args.use_xyz else 2
    model = STGCN(num_class=args.num_class, in_channels=in_channels, A=A).to(device)

    # Data
    train_ds = KeypointDataset(args.labels_csv, split="train", use_xyz=args.use_xyz)
    val_ds = KeypointDataset(args.labels_csv, split="val", use_xyz=args.use_xyz)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # Optimizer + loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    # LR schedule: drop at epoch 20 and 40 by 0.1
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 40],
        gamma=0.1
    )

    # Checkpointing
    os.makedirs(args.save_dir, exist_ok=True)
    best_val = 0.0
    best_path = os.path.join(args.save_dir, "best_stgcn.pt")

    for epoch in range(1, args.epochs + 1):
        # ---- TRAIN ----
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", ncols=90):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)

        # ---- VAL ----
        model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                v_correct += (pred == y).sum().item()
                v_total += x.size(0)

        val_acc = v_correct / max(1, v_total)

        # Print current LR
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}: lr={current_lr:.6f} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # Save best
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model": model.state_dict(),
                # "use_xy": args.use_xy,
                "num_class": args.num_class,
                "in_channels": in_channels
            }, best_path)
            print(f"✅ Saved best checkpoint: {best_path} (val_acc={best_val:.4f})")

        # Step scheduler at end of epoch
        scheduler.step()

    print("Training done. Best val_acc:", best_val)


if __name__ == "__main__":
    main()
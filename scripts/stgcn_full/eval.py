import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from graph_hand import Graph
from dataset import KeypointDataset
from model import STGCN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", type=str, default="dataset/processed/labels_clean.csv")
    ap.add_argument("--ckpt", type=str, default="checkpoints/best_stgcn.pt")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    # use_xyz = ckpt["use_xyz"]
    in_channels = ckpt["in_channels"]
    num_class = ckpt["num_class"]

    # Graph + Model
    graph = Graph(strategy="spatial", max_hop=1)
    A = graph.A
    model = STGCN(num_class=num_class, in_channels=in_channels, A=A).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Data
    test_ds = KeypointDataset(args.labels_csv, split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # Predict
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metrics
    overall_acc = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    print(f"Macro F1-score : {macro_f1:.4f}\n")

    # Per-class accuracy
    per_class_acc = []
    for c in range(num_class):
        mask = (y_true == c)
        if mask.sum() == 0:
            acc_c = np.nan
        else:
            acc_c = (y_pred[mask] == c).mean()
        per_class_acc.append(acc_c)

    # Save per-class accuracy CSV
    per_class_path = os.path.join(args.out_dir, "per_class_accuracy.csv")
    with open(per_class_path, "w", encoding="utf-8") as f:
        f.write("class_id,accuracy\n")
        for c, acc_c in enumerate(per_class_acc):
            if np.isnan(acc_c):
                f.write(f"{c},\n")
            else:
                f.write(f"{c},{acc_c:.6f}\n")
    print("Saved:", per_class_path)

    # Classification report
    report = classification_report(y_true, y_pred, digits=4)
    report_path = os.path.join(args.out_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Overall Accuracy: {overall_acc:.6f}\n")
        f.write(f"Macro F1-score : {macro_f1:.6f}\n\n")
        f.write(report)
    print("Saved:", report_path)
    print("\nClassification report:\n")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.colorbar()
    plt.tight_layout()

    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    print("Saved:", cm_path)

    # Optionally show on screen (comment out if you don't want pop-up)
    # plt.show()


if __name__ == "__main__":
    main()
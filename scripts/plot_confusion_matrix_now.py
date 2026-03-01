import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.stgcn_full.dataset import KeypointDataset
from scripts.stgcn_full.model import STGCN
from scripts.stgcn_full.graph_hand import Graph

CKPT_PATH = "checkpoints/best_stgcn.pt"
LABELS_CSV = "dataset/processed/labels_clean.csv"

CLASS_NAMES = [
    "А-A","Б-B","В-V","Г-G","Д-D","Е-YE","Ё-YO","Ж-J","З-Z","И-I",
    "Й-hI","К-K","Л-L","М-M","Н-N","О-O","Ө-OU","П-P","Р-R","С-S",
    "Т-T","У-U","Ү-Y","Х-H","Ф-F","Ц-TS","Ч-CH","Ш-SH","Щ-SHCH",
    "Ъ-Htemdeg","Ы-ERII","Ь-Ztemdeg","Э-E","Ю-YU","Я-YA"
]

def main():
    Path("results").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT_PATH, map_location=device)

    graph = Graph(strategy="spatial", max_hop=1)
    A = graph.A

    model = STGCN(num_class=len(CLASS_NAMES), in_channels=ckpt["in_channels"], A=A).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_ds = KeypointDataset(LABELS_CSV, split="test")
    loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    # Optional: row-normalized (percent) for thesis readability
    cm_norm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    # Plot (normalized)
    plt.figure(figsize=(14, 12))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title("Confusion Matrix (row-normalized)")
    plt.colorbar(fraction=0.046, pad=0.04)

    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=90, fontsize=7)
    plt.yticks(ticks, CLASS_NAMES, fontsize=7)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_norm.png", dpi=300, bbox_inches="tight")
    print("Saved: results/confusion_matrix_norm.png")

    # Plot (raw counts) too
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (counts)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(ticks, CLASS_NAMES, rotation=90, fontsize=7)
    plt.yticks(ticks, CLASS_NAMES, fontsize=7)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_counts.png", dpi=300, bbox_inches="tight")
    print("Saved: results/confusion_matrix_counts.png")

if __name__ == "__main__":
    main()
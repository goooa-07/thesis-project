import sys
from pathlib import Path

# Add project root to Python path so imports work
ROOT = Path(__file__).resolve().parents[1]   # thesis_code/
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import classification_report

# Import using the actual folder path
from scripts.stgcn_full.dataset import KeypointDataset
from scripts.stgcn_full.model import STGCN
from scripts.stgcn_full.graph_hand import Graph


CLASS_NAMES = [
"А-A","Б-B","В-V","Г-G","Д-D","Е-YE","Ё-YO","Ж-J","З-Z","И-I",
"Й-hI","К-K","Л-L","М-M","Н-N","О-O","Ө-OU","П-P","Р-R","С-S",
"Т-T","У-U","Ү-Y","Х-H","Ф-F","Ц-TS","Ч-CH","Ш-SH","Щ-SHCH",
"Ъ-Htemdeg","Ы-ERII","Ь-Ztemdeg","Э-E","Ю-YU","Я-YA"
]

ckpt_path = "checkpoints/best_stgcn.pt"
labels_csv = "dataset/processed/labels_clean.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(ckpt_path, map_location=device)

graph = Graph(strategy="spatial", max_hop=1)
A = graph.A

model = STGCN(
    num_class=len(CLASS_NAMES),
    in_channels=ckpt["in_channels"],
    A=A
).to(device)

model.load_state_dict(ckpt["model"])
model.eval()

test_ds = KeypointDataset(labels_csv, split="test")
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(pred)
        y_true.extend(y.tolist())

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    digits=4
)

df = pd.DataFrame(report).transpose()
Path("results").mkdir(exist_ok=True)
df.to_csv("results/named_classification_report.csv", encoding="utf-8-sig")

print("✅ Saved: results/named_classification_report.csv")
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

class KeypointDataset(Dataset):
    def __init__(self, labels_csv, split="train", use_xyz=False):
        self.items = []
        self.use_xyz = use_xyz

        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.items.append((row["path"], int(row["label"])))

        if not self.items:
            raise ValueError(f"No rows for split='{split}' in {labels_csv}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        kp = np.load(path)  # (T,21,3)

        kp = kp[:, :, :3] if self.use_xyz else kp[:, :, :2]
        x = np.transpose(kp, (2, 0, 1)).astype(np.float32)  # (C,T,V)

        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)
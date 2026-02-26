import csv
from pathlib import Path

labels_csv = Path("dataset/processed/labels.csv")
out_csv = Path("dataset/processed/labels_clean.csv")

project_root = Path(".").resolve()

kept = 0
dropped = 0

with open(labels_csv, "r", encoding="utf-8") as f_in, open(out_csv, "w", newline="", encoding="utf-8") as f_out:
    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames
    if fieldnames is None:
        raise SystemExit("labels.csv has no header")

    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        p = Path(row["path"])
        if not p.is_absolute():
            p = (project_root / p).resolve()

        if p.exists():
            row["path"] = str(p)
            writer.writerow(row)
            kept += 1
        else:
            dropped += 1

print("✅ Wrote:", out_csv)
print("Kept:", kept, "Dropped:", dropped)
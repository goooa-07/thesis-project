import glob
import numpy as np

files = glob.glob("dataset/processed/keypoints/**/*.npy", recursive=True)
print("Found npy:", len(files))
assert len(files) > 0, "No keypoints found."

a = np.load(files[0])
print("Example file:", files[0])
print("Shape:", a.shape)
print("Min/Max:", a.min(), a.max())
print("NaNs:", np.isnan(a).sum())
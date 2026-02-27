import time
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.stgcn_full.model import STGCN
from scripts.stgcn_full.graph_hand import Graph

CKPT_PATH = "checkpoints/best_stgcn.pt"
TARGET_FPS = 30
CLIP_SECONDS = 3
T_TARGET = TARGET_FPS * CLIP_SECONDS  # 90

CLASS_NAMES = [
    "А-A","Б-B","В-V","Г-G","Д-D","Е-YE","Ё-YO","Ж-J","З-Z","И-I",
    "Й-hI","К-K","Л-L","М-M","Н-N","О-O","Ө-OU","П-P","Р-R","С-S",
    "Т-T","У-U","Ү-Y","Х-H","Ф-F","Ц-TS","Ч-CH","Ш-SH","Щ-SHCH",
    "Ъ-Htemdeg","Ы-ERII","Ь-Ztemdeg","Э-E","Ю-YU","Я-YA"
]

HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

def load_model_xy(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    graph = Graph(strategy="spatial", max_hop=1)
    A = graph.A

    model = STGCN(num_class=len(CLASS_NAMES), in_channels=2, A=A).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, device

def resample_to_T(seq, T_target):
    T, V, C = seq.shape
    if T == T_target:
        return seq
    idx = np.linspace(0, T - 1, T_target)
    idx0 = np.floor(idx).astype(int)
    idx1 = np.clip(idx0 + 1, 0, T - 1)
    w = idx - idx0
    out = (1 - w)[:, None, None] * seq[idx0] + w[:, None, None] * seq[idx1]
    return out.astype(np.float32)

def normalize_like_training(seq_xy):
    """
    IMPORTANT: Many keypoint pipelines normalize by:
    - subtract wrist (joint 0)
    - divide by hand scale (e.g., distance wrist->middle_mcp joint 9)
    If your training used this, real-time MUST match.

    This version does a safe standard normalization:
    center by wrist and scale by mean bone length.
    """
    seq = seq_xy.copy()  # (T,21,2)

    # center by wrist (joint 0)
    wrist = seq[:, 0:1, :]  # (T,1,2)
    seq = seq - wrist

    # scale by average distance from wrist to all joints (avoid divide by zero)
    scale = np.linalg.norm(seq, axis=2).mean(axis=1)  # (T,)
    scale = np.maximum(scale, 1e-6)
    seq = seq / scale[:, None, None]

    return seq.astype(np.float32)

def predict(model, device, seq_xy):
    # seq_xy: (T,21,2)
    x = np.transpose(seq_xy, (2, 0, 1)).astype(np.float32)  # (2,T,21)
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0]
        pred = int(prob.argmax().item())
        conf = float(prob[pred].item())
    return pred, conf

def draw_keypoints(frame, kps_xy_norm):
    """
    kps_xy_norm: (21,2) normalized [0..1]
    Draws joints + edges.
    """
    h, w = frame.shape[:2]
    pts = []
    for x, y in kps_xy_norm:
        px, py = int(x * w), int(y * h)
        pts.append((px, py))
        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
    for a, b in HAND_EDGES:
        cv2.line(frame, pts[a], pts[b], (0, 255, 255), 2)

def main():
    assert Path(CKPT_PATH).exists(), f"Checkpoint not found: {CKPT_PATH}"
    model, device = load_model_xy(CKPT_PATH)
    print("Model loaded on", device)

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found")

    recording = False
    frames = []
    kp_seq = []
    detected_frames = 0
    start_time = 0.0
    message = "SPACE: record 3s | ESC: quit"
    last_conf = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # If your training videos were mirrored, uncomment next line:
        # frame = cv2.flip(frame, 1)

        # Run MediaPipe every frame so we can visualize detection live
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        kps_xy = None
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            kps_xy = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
            draw_keypoints(frame, kps_xy)

        # Overlay info
        status = "HAND: OK" if kps_xy is not None else "HAND: NOT DETECTED"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 255, 0) if kps_xy is not None else (0, 0, 255), 2)
        cv2.putText(frame, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if recording:
            frames.append(frame.copy())
            if kps_xy is not None:
                kp_seq.append(kps_xy)
                detected_frames += 1
            else:
                # store zeros (so we can measure detection rate)
                kp_seq.append(np.zeros((21, 2), dtype=np.float32))

            elapsed = time.time() - start_time
            cv2.putText(frame, f"REC {elapsed:.1f}s  detected:{detected_frames}/{len(kp_seq)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if elapsed >= CLIP_SECONDS:
                recording = False

                seq = np.stack(kp_seq, axis=0)  # (T,21,2)
                seq = resample_to_T(seq, T_TARGET)

                detect_ratio = detected_frames / max(1, len(kp_seq))

                # Debug stats BEFORE normalization
                std_raw = float(seq.std())
                mean_raw = float(seq.mean())

                # Apply normalization (match training as needed)
                seq_norm = normalize_like_training(seq)

                # Debug stats AFTER normalization
                std_norm = float(seq_norm.std())
                mean_norm = float(seq_norm.mean())

                # If hand was not detected enough -> prediction is unreliable
                if detect_ratio < 0.7:
                    message = f"Too many missing hands ({detect_ratio*100:.0f}%). Try better lighting/background."
                else:
                    pred, conf = predict(model, device, seq_norm)
                    last_conf = conf
                    message = f"Pred: {CLASS_NAMES[pred]}  conf={conf:.3f}  det={detect_ratio*100:.0f}%"

                print(
                    f"[DEBUG] detect_ratio={detect_ratio:.2f} "
                    f"raw(mean={mean_raw:.4f}, std={std_raw:.4f}) "
                    f"norm(mean={mean_norm:.4f}, std={std_norm:.4f}) "
                    f"last_conf={last_conf:.3f}"
                )

                frames = []
                kp_seq = []
                detected_frames = 0

        cv2.imshow("ST-GCN Real-time DEBUG (XY)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32 and not recording:
            recording = True
            frames = []
            kp_seq = []
            detected_frames = 0
            start_time = time.time()
            message = "Recording..."

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
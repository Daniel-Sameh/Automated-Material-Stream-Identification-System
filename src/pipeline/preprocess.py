import cv2
import numpy as np

def gray_world_white_balance(bgr: np.ndarray) -> np.ndarray:
    """Simple, fast white balance to reduce webcam AWB drift."""
    img = bgr.astype(np.float32)
    b, g, r = cv2.split(img)

    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
    gray = (b_mean + g_mean + r_mean) / 3.0

    b *= (gray / (b_mean + 1e-6))
    g *= (gray / (g_mean + 1e-6))
    r *= (gray / (r_mean + 1e-6))

    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

def clahe_on_l_channel(bgr: np.ndarray) -> np.ndarray:
    """Stabilize illumination: CLAHE on Lab L channel."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def preprocess_for_inference(bgr: np.ndarray) -> np.ndarray:
    bgr = gray_world_white_balance(bgr)
    bgr = clahe_on_l_channel(bgr)
    return bgr
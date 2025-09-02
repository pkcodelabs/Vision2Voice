import cv2
import numpy as np
import pytesseract
from typing import Tuple

# If tesseract is not in PATH, set the path explicitly, e.g. on Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(img: np.ndarray, max_width: int = 1600) -> np.ndarray:
    # Resize (keep aspect ratio) to speed up and improve OCR
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bilateral filter preserves edges while reducing noise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # adaptive threshold to handle uneven lighting
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 41, 11)
    # optional morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return opened

def deskew(image: np.ndarray) -> np.ndarray:
    # deskew using image moments / minAreaRect on text-like image
    coords = np.column_stack(np.where(image < 255))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def extract_text_pytesseract(image_path: str, lang: str = 'eng') -> Tuple[str, np.ndarray]:
    """
    Returns (extracted_text, processed_image_for_debug)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    pre = preprocess_image(img)
    pre = deskew(pre)

    # Tesseract config: use LSTM engine (oem 1), automatic page segmentation mode (psm 3 or 6)
    custom_config = r'--oem 1 --psm 6'  # psm 6 = assume a single uniform block of text
    text = pytesseract.image_to_string(pre, lang=lang, config=custom_config)

    return text.strip(), pre

# Example usage
if __name__ == "__main__":
    img_path = "sample_sign.jpg"
    text, debug_img = extract_text_pytesseract(img_path)
    print("Detected text:\n", text)
    # cv2.imshow("preprocessed", debug_img); cv2.waitKey(0)


import cv2

def draw_boxes(image_path: str, ocr_results, out_path: str = "boxed.jpg"):
    img = cv2.imread(image_path)
    for bbox, text, conf in ocr_results:
        # bbox is list of 4 points
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
        # place text above box
        x, y = pts[0]
        cv2.putText(img, text, (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imwrite(out_path, img)

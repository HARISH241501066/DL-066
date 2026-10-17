import cv2
import numpy as np

def assess_image_quality(image_path: str) -> dict:
    """
    Analyzes the uploaded MRI image for blur, contrast, and noise.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur Detection (Variance of Laplacian)
    # Higher variance means more edges (less blur).
    variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize loosely to a 0-100 scale (this threshold depends on image type, 
    # but for prototype we use 1000 as a reference for 'very sharp').
    sharpness_score = min(100, int((variance_of_laplacian / 1000.0) * 100))
    if sharpness_score < 10:
        sharpness_score = 10 # Floor it for presentation
    
    # 2. Contrast Check (RMS contrast)
    rms_contrast = gray.std()
    # Normalize to 0-100 (standard deviation can theoretically go up to 127.5 max for 8-bit image)
    contrast_score = min(100, int((rms_contrast / 100.0) * 100))
    if contrast_score < 10:
        contrast_score = 10
        
    # 3. Noise Estimation
    # A simple approach is using blur difference or standard deviation of smooth areas.
    # For prototype, we'll invert sharpness in a way or use a basic filtering approach.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_diff = cv2.absdiff(gray, blurred)
    noise_level_val = noise_diff.mean()
    
    if noise_level_val > 15:
        noise_level = "High"
    elif noise_level_val > 5:
        noise_level = "Medium"
    else:
        noise_level = "Low"
        
    # Overall Quality
    if sharpness_score > 60 and contrast_score > 50 and noise_level in ["Low", "Medium"]:
        overall_quality = "GOOD"
    elif sharpness_score < 30 or contrast_score < 30:
        overall_quality = "POOR"
    else:
        overall_quality = "FAIR"

    return {
        "sharpness": sharpness_score,
        "contrast": contrast_score,
        "noise_level": noise_level,
        "overall_quality": overall_quality,
        "details": "Motion blur detected." if overall_quality == "POOR" and sharpness_score < 30 else "Prediction reliability may be reduced." if overall_quality == "POOR" else ""
    }

import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import datetime

def generate_pdf_report(original_image_path: str, quality_metrics: dict, inference_results: dict) -> str:
    """
    Generates a PDF medical report for Q-MED VISION.
    """
    output_filename = f"report_{os.path.basename(original_image_path)}.pdf"
    output_path = os.path.join(os.path.dirname(original_image_path), output_filename)
    
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Q-MED VISION")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, "AI-ASSISTED MEDICAL IMAGE ANALYSIS")
    
    # Metadata
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(50, height - 100, f"Patient ID: Anonymous")
    c.drawString(50, height - 120, f"Analysis Date: {now}")
    
    # Quality Metrics
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 150, "Image Quality Assessment:")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 170, f"Overall Quality: {quality_metrics['overall_quality']}")
    c.drawString(50, height - 190, f"Sharpness: {quality_metrics['sharpness']}/100")
    c.drawString(50, height - 210, f"Contrast: {quality_metrics['contrast']}/100")
    c.drawString(50, height - 230, f"Noise Level: {quality_metrics['noise_level']}")
    
    # Prediction Results
    c.setFont("Helvetica-Bold", 12)
    c.drawString(300, height - 150, "AI Diagnosis Results:")
    c.setFont("Helvetica", 12)
    c.drawString(300, height - 170, f"Prediction: {inference_results['prediction']}")
    c.drawString(300, height - 190, f"Probability: {inference_results['probability']}")
    c.drawString(300, height - 210, f"Confidence: {inference_results['confidence']}")
    c.drawString(300, height - 230, f"Risk Indicator: {inference_results['risk_level']}")
    
    # Model info
    c.drawString(50, height - 260, f"Model: {inference_results.get('model_name', 'unknown')} (research prototype - not for clinical use)")
    
    # Images (Original and Heatmap)
    y_pos = height - 500
    
    try:
        if os.path.exists(original_image_path):
            img = ImageReader(original_image_path)
            c.drawImage(img, 50, y_pos, width=200, height=200, preserveAspectRatio=True)
            c.drawString(100, y_pos - 20, "Original MRI")
            
        heatmap_path = os.path.join(os.path.dirname(original_image_path), inference_results.get("heatmap_image", ""))
        if inference_results.get("heatmap_image") and os.path.exists(heatmap_path):
            heatmap_img = ImageReader(heatmap_path)
            c.drawImage(heatmap_img, 300, y_pos, width=200, height=200, preserveAspectRatio=True)
            c.drawString(350, y_pos - 20, "AI Heatmap (Grad-CAM)")
    except Exception as e:
        print(f"Warning: Could not embed images in PDF: {e}")
        
    # Disclaimer
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 100, "Disclaimer:")
    c.drawString(50, 85, "This system is intended for research and clinical decision-support demonstration only.")
    c.drawString(50, 70, "It does not replace diagnosis by a qualified medical professional.")
    
    c.save()
    
    return output_filename

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os

from services.image_quality import assess_image_quality
from services.inference import process_and_infer
from services.report_generator import generate_pdf_report

app = FastAPI(title="Q-MED VISION API")
API_BASE_URL = "http://localhost:8000"
# Trigger reload

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Save file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Assess Image Quality
        quality_metrics = assess_image_quality(file_path)
        
        # 2. Run Inference (Classical + Simulated Quantum) & Grad-CAM
        inference_results = process_and_infer(file_path)
        
        # 3. Generate PDF Report
        report_path = generate_pdf_report(file_path, quality_metrics, inference_results)
        
        # Modify heatmap_image to be a URL path
        if inference_results.get("heatmap_image"):
            inference_results["heatmap_url"] = f"{API_BASE_URL}/uploads/{inference_results['heatmap_image']}"
        else:
            inference_results["heatmap_url"] = None

        # Return all results
        return JSONResponse(content={
            "quality": quality_metrics,
            "inference": inference_results,
            "report_url": f"{API_BASE_URL}/uploads/{report_path}"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import os

from fastapi import FastAPI,UploadFile,File
from fastapi.responses import FileResponse
import shutil
from tracking import process_video
from fastapi.middleware.cors import CORSMiddleware
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    input_path= f"input_{file.filename}"
    output_path= f"output_{file.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    process_video(input_path, output_path)
    if not os.path.exists(output_path):
        return {"error": "Output file not created"}
    return FileResponse(output_path, media_type="video/mp.4",filename="outpat.mp4")

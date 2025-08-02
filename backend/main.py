# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.recognition_api import router as api_router

app = FastAPI(
    title="Sign Language Translator API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- KẾT THÚC SỬA ĐỔI ---

@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với API Nhận dạng Thủ ngữ!"}

app.include_router(api_router, prefix="/api")
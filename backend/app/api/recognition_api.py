# app/api/recognition_api.py (Phiên bản đã sửa lỗi API Key)

import uuid
import base64
import cv2
import numpy as np
import os
import google.generativeai as genai 
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from app.services.recognition_service import SignRecognizer, SentenceBuilder
from fastapi.responses import FileResponse
from gtts import gTTS


try:
    
    MY_API_KEY = "AIzaSyAnjnZYp1YohxUL2OhNslSdBeGu6-baRuk" 
    
    genai.configure(api_key=MY_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print(">>> Gemini đã được cấu hình thành công bằng API Key trong code. <<<")
except Exception as e:
    print(f"Lỗi khi cấu hình Gemini: {e}")
    gemini_model = None


router = APIRouter()

recognizer = SignRecognizer(model_path="ml_models/best_model_landmark_with_unknown.keras")


@router.websocket("/ws/recognize")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client đã kết nối.")
    sentence_builder = SentenceBuilder()
    try:
        while True:
            base64_data = await websocket.receive_text()
            
            header, encoded = base64_data.split(",", 1)
            nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                predicted_char, drawn_frame = recognizer.predict(frame)
                
                _, buffer_drawn = cv2.imencode('.jpg', drawn_frame)
                drawn_frame_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer_drawn).decode("utf-8")

                sentence_builder.add_char(predicted_char)
                
                payload = {
                    "video_frame": drawn_frame_base64,
                    "sentence": sentence_builder.get_sentence()
                }
                
                await websocket.send_json(payload)
                
    except WebSocketDisconnect:
        print("Client đã ngắt kết nối.")
    except Exception as e:
        print(f"Lỗi xảy ra trong WebSocket: {e}")

# ENDPOINT ĐỂ XỬ LÝ VÀ DỊCH CÂU

class SentenceRequest(BaseModel):
    raw_sentence: str
    target_language: str

async def process_with_gemini(prompt: str):
    """Hàm trợ giúp để gọi Gemini API và xử lý lỗi."""
    if not gemini_model:
        raise HTTPException(status_code=500, detail="Gemini API chưa được cấu hình.")
    try:
        response = await gemini_model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        # Thêm dòng print này để dễ dàng gỡ lỗi hơn trong tương lai
        print(f"!!! LỖI CHI TIẾT TỪ GEMINI API: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi gọi Gemini API: {e}")

@router.post("/process-sentence")
async def process_sentence_endpoint(request: SentenceRequest):
    """
    Nhận câu thô, sửa lỗi bằng Gemini, sau đó dịch sang ngôn ngữ đích.
    """
    if not request.raw_sentence:
        return {"corrected_sentence": "", "translated_sentence": ""}

    prompt_fix = f"""Hãy sửa lỗi chính tả, thêm dấu câu và định dạng lại câu tiếng Việt sau đây cho đúng ngữ pháp. Câu này được tạo ra từ hệ thống nhận diện ngôn ngữ ký hiệu nên có thể có nhiều lỗi. Chỉ trả về duy nhất câu đã hoàn chỉnh, không giải thích gì thêm.
Câu gốc: '{request.raw_sentence}'"""
    
    corrected_sentence = await process_with_gemini(prompt_fix)

    translated_sentence = corrected_sentence
    if request.target_language.lower() != "vietnamese":
        prompt_translate = f"""Dịch chính xác câu sau đây từ tiếng Việt sang tiếng {request.target_language}. Chỉ trả về duy nhất kết quả đã dịch, không giải thích gì thêm.
Câu cần dịch: '{corrected_sentence}'"""
        translated_sentence = await process_with_gemini(prompt_translate)

    return {
        "corrected_sentence": corrected_sentence,
        "translated_sentence": translated_sentence,
    }

# ENDPOINT CHUYỂN VĂN BẢN THÀNH GIỌNG NÓI
class SpeakRequest(BaseModel):
    text: str
    lang: str = "vi" # Đặt mặc định là tiếng Việt

@router.post("/speak")
async def speak_text(request: SpeakRequest):
    # Tạo thư mục tạm nếu chưa có
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        tts = gTTS(text=request.text, lang=request.lang, slow=False)
        # Tạo tên file duy nhất trong thư mục tạm
        filename = os.path.join(temp_dir, f"speech_{uuid.uuid4().hex}.mp3")
        tts.save(filename)
        # Trả về file và FastAPI sẽ tự động xóa nó sau khi gửi
        return FileResponse(path=filename, media_type="audio/mpeg", filename="speech.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo giọng nói: {e}")
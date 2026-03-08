import os
import sqlite3
import shutil
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List
from datetime import datetime, timezone
from openai import AsyncOpenAI
from dotenv import load_dotenv
from transformers import pipeline

# Load API Key
load_dotenv()

# Cấu hình OpenAI
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

AI_MODEL = "gpt-4o-mini"
DB_NAME = "styles.db"  # Tên file database sẽ được tạo ra

app = FastAPI(title="AI Copywriting Service")


# --- 1. DATA MODELS ---
class Style(BaseModel):
    name: str
    description: str


class GenRequest(BaseModel):
    style: str
    product_description: str


class GenResponse(BaseModel):
    original_description: str
    style: str
    generated_description: str
    generated_at: str


# --- 2. DATABASE SETUP ---
def init_db():
    """Khởi tạo database và bảng nếu chưa có"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Tạo bảng styles
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS styles
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       name
                       TEXT
                       UNIQUE
                       NOT
                       NULL,
                       description
                       TEXT
                       NOT
                       NULL
                   )
                   ''')

    # Kiểm tra xem có dữ liệu chưa, nếu chưa thì nạp dữ liệu mẫu
    cursor.execute('SELECT count(*) FROM styles')
    if cursor.fetchone()[0] == 0:
        default_data = [
            ("văn minh", "Phong cách lịch sự, trang trọng, sử dụng từ ngữ chuẩn mực"),
            ("văn hoá người tày", "Kết hợp yếu tố văn hóa dân tộc Tày, sử dụng hình ảnh và từ ngữ đặc trưng"),
            ("tan", "Phong cách cá nhân hóa của người dùng"),
            ("chuyên nghiệp", "Ngôn ngữ nghiệp vụ, rõ ràng, tập trung vào thông số kỹ thuật"),
            ("thân thiện", "Giọng điệu gần gũi, ấm áp, dễ tiếp cận với khách hàng")
        ]
        cursor.executemany('INSERT INTO styles (name, description) VALUES (?, ?)', default_data)
        conn.commit()
        print(">>> Đã khởi tạo dữ liệu mẫu vào SQLite.")

    conn.close()


# Chạy hàm khởi tạo ngay khi file được load
init_db()


# --- 3. ENDPOINTS ---

@app.get("/styles", response_model=List[Style])
async def get_styles():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT name, description FROM styles")
    rows = cursor.fetchall()

    conn.close()

    # Chuyển đổi từng row (sqlite3.Row) thành dict chuẩn
    return [dict(row) for row in rows]


@app.post("/add-style")
async def add_style(style: Style):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        cursor.execute("INSERT INTO styles (name, description) VALUES (?, ?)", (style.name, style.description))
        conn.commit()
        conn.close()

        return {"message": "Style added successfully", "data": style}
    except sqlite3.IntegrityError:
        # Bắt lỗi nếu trùng tên style (do cột name là UNIQUE)
        raise HTTPException(status_code=400, detail="Style name already exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gen", response_model=GenResponse)
async def generate_description(request: GenRequest):
    # 1. Lấy context từ SQLite
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Query lấy description dựa trên tên style
    cursor.execute("SELECT description FROM styles WHERE name = ?", (request.style,))
    row = cursor.fetchone()
    conn.close()

    style_context = "Viết sáng tạo, tự nhiên"  # Fallback
    if row:
        style_context = row[0]  # Lấy cột description

    # 2. Tạo Prompt
    messages = [
        {
            "role": "system",
            "content": "Bạn là một chuyên gia Copywriter. Nhiệm vụ của bạn là viết lại mô tả sản phẩm dựa trên phong cách được yêu cầu."
        },
        {
            "role": "user",
            "content": (
                f"Thông tin sản phẩm gốc: '{request.product_description}'\n"
                f"Phong cách yêu cầu: '{request.style}'\n"
                f"Đặc tả phong cách này: {style_context}\n\n"
                f"Yêu cầu: Hãy viết lại mô tả sản phẩm trên theo đúng phong cách đã chọn. "
                f"Chỉ trả về nội dung văn bản kết quả, không bao gồm lời dẫn hay dấu ngoặc kép."
            )
        }
    ]

    try:
        # 3. Gọi OpenAI API
        response = await client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        generated_text = response.choices[0].message.content.strip()

        # 4. Trả về kết quả
        return GenResponse(
            original_description=request.product_description,
            style=request.style,
            generated_description=generated_text,
            generated_at=datetime.now(timezone.utc).isoformat()
        )

    except Exception as e:
        print(f"OpenAI Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- KHỞI TẠO MODEL STT (Load 1 lần khi chạy server) ---
print("Đang tải model PhoWhisper... Vui lòng đợi...")
stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-medium")
print("Tải model thành công!")


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    # 1. Kiểm tra định dạng file (hỗ trợ .wav, .mp3, .flac, .ogg)
    allowed_extensions = ["wav", "mp3", "flac", "ogg"]
    file_ext = file.filename.split(".")[-1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400,
                            detail="Định dạng file không được hỗ trợ. Chỉ hỗ trợ .wav, .mp3, .flac, .ogg")

    # 2. Lưu file audio tạm thời xuống ổ đĩa để đưa vào model
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 3. Tích hợp model PhoWhisper tại đây
        result = stt_pipeline(temp_file_path)
        generated_text = result["text"]

        # 4. Xóa file tạm sau khi xử lý xong
        os.remove(temp_file_path)

        # Trả về response đúng định dạng
        return {"data": generated_text}

    except Exception as e:
        # Nhớ xóa file tạm nếu có lỗi xảy ra
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        print(f"STT Error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi khi xử lý âm thanh")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5001)))

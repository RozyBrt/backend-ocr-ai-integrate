import os
import base64
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load variables dari .env
load_dotenv()

app = FastAPI(title="OCR Vision Bridge API")

# Izinkan CORS agar Flutter tidak terblokir
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-ocr")
async def process_ocr(file: UploadFile = File(...)):
    print(f"-> Menerima file: {file.filename}, Tipe Konten: {file.content_type}")
    
    # Validasi dilonggarkan agar tidak gampang Error 400
    if not file.content_type:
        print("!! Warning: Tipe konten kosong, mencoba memproses saja...")
    elif not file.content_type.startswith("image/"):
        print(f"!! Warning: Tipe konten '{file.content_type}' bukan image, tetap mencoba memproses...")
    
    # 2. Baca file dan konversi ke Base64
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')

    # 3. Ambil konfigurasi dari .env
    api_key = os.environ.get("SUMOPOD_API_KEY", "").strip()
    sumopod_url = os.environ.get("SUMOPOD_URL", "https://ai.sumopod.com/v1/chat/completions")
    sumopod_model = os.environ.get("SUMOPOD_VISION_MODEL", "claude-haiku-4-5")

    if not api_key:
        raise HTTPException(status_code=500, detail="API Key tidak ditemukan di .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Payload sederhana format OpenAI Vision
    payload = {
        "model": sumopod_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Ekstrak seluruh teks dalam gambar ini secara akurat. Langsung berikan hasilnya saja tanpa kata pengantar."},
                    {
                        "type": "image_url",
                        "image_url": {
                            # Paksa media_type ke image/jpeg agar SumoPod tidak komplain
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2000
    }
    
    # 4. Kirim request ke SumoPod
    try:
        response = requests.post(sumopod_url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            return JSONResponse(
                content={"error": "Gagal dari SumoPod", "detail": response.text}, 
                status_code=response.status_code
            )

        result = response.json()
        hasil_teks = result["choices"][0]["message"]["content"]
        
        return JSONResponse({"hasil": hasil_teks.strip()})
        
    except Exception as e:
        return JSONResponse(content={"error": "Terjadi kesalahan internal", "detail": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # Menjalankan server di port 8000
    uvicorn.run("ocr_ai:app", host="0.0.0.0", port=8000, reload=True)
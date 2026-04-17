import os
import base64
import requests
import gc
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Konfigurasi Hardcoded (Lokal/Ngrok Mode)
SUMOPOD_API_KEY = "sk-U8jMown1Umhd_53gPH8GwA"
SUMOPOD_URL = "https://ai.sumopod.com/v1/chat/completions"
SUMOPOD_MODEL = "claude-haiku-4-5"

app = FastAPI(title="OCR AI Bridge - Lokal")

# CORS tetap diizinkan agar Flutter tidak terblokir saat testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-ocr")
async def process_ocr(file: UploadFile = File(...)):
    print(f"-> Menerima file: {file.filename}")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
    
    try:
        # Baca dan encode gambar
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        del contents
        gc.collect()

        headers = {
            "Authorization": f"Bearer {SUMOPOD_API_KEY}",
            "Content-Type": "application/json"
        }

        # Instruksi OCR Strict
        payload = {
            "model": SUMOPOD_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Ekstrak semua teks dari gambar ini secara akurat. Jangan tambahkan kata pengantar atau analisis. HANYA hasil teks OCR saja."},
                        {"type": "image_url", "image_url": {"url": f"data:{file.content_type};base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 2000
        }
        
        # Kirim ke SumoPod
        response = requests.post(SUMOPOD_URL, json=payload, headers=headers, timeout=60)
        
        # Cleanup Memori
        del base64_image
        del payload
        gc.collect()

        if response.status_code != 200:
            return JSONResponse(content={"error": "SumoPod Error", "detail": response.text}, status_code=response.status_code)

        result = response.json()
        hasil_teks = result["choices"][0]["message"]["content"].strip()
        
        # Potong teks sampah jika ada
        if "---" in hasil_teks: hasil_teks = hasil_teks.split("---")[0]
        if "Namun, saya catatan" in hasil_teks: hasil_teks = hasil_teks.split("Namun, saya catatan")[0]
        
        print("-> OCR Berhasil!")
        return JSONResponse({"hasil": hasil_teks.strip()})
        
    except Exception as e:
        print(f"-> Error: {str(e)}")
        return JSONResponse(content={"error": "System Error", "detail": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # Jalankan di localhost port 8000
    print("-> Server OCR Lokal Berjalan...")
    uvicorn.run("ocr_ai:app", host="0.0.0.0", port=8000, reload=True)
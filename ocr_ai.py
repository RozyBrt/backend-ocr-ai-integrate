import os
import base64
import requests
import gc
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load variabels dari .env
load_dotenv()

app = FastAPI(title="OCR & Summary Dual-Brain API")

# [CORS POLICY]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. ENDPOINT OCR (Spesialis: Claude)
# ---------------------------------------------------------
@app.post("/process-ocr")
async def process_ocr(file: UploadFile = File(...)):
    print(f"-> [OCR] Menerima file: {file.filename}")
    
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
    
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        del contents
        gc.collect()

        # Baca config
        load_dotenv(override=True)
        api_key = os.getenv("SUMOPOD_API_KEY", "").strip()
        ocr_model = os.getenv("OCR_MODEL", "claude-haiku-4-5")
        sumopod_url = os.getenv("SUMOPOD_URL", "https://ai.sumopod.com/v1/chat/completions")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Ekstrak semua teks dari gambar ini secara akurat. Langsung berikan hasil teksnya saja tanpa kata pengantar."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 2000
        }
        
        response = requests.post(sumopod_url, json=payload, headers=headers, timeout=60)
        
        # Cleanup
        del base64_image
        del payload
        gc.collect()

        if response.status_code != 200:
            return JSONResponse(content={"error": "OCR Model Error", "detail": response.text}, status_code=response.status_code)

        result = response.json()
        hasil_teks = result["choices"][0]["message"]["content"].strip()
        print("-> OCR Berhasil!")
        return JSONResponse({"hasil": hasil_teks})
        
    except Exception as e:
        return JSONResponse(content={"error": "OCR System Error", "detail": str(e)}, status_code=500)

# ---------------------------------------------------------
# 2. ENDPOINT SUMMARIZE (Spesialis: GPT-4o-mini)
# ---------------------------------------------------------
class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
async def summarize_text(req: SummarizeRequest):
    print("-> [SUMMARY] Menerima request ringkasan")
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Teks kosong bray!")
    
    try:
        load_dotenv(override=True)
        api_key = os.getenv("SUMOPOD_API_KEY", "").strip()
        summary_model = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")
        sumopod_url = os.getenv("SUMOPOD_URL", "https://ai.sumopod.com/v1/chat/completions")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # System prompt agar ringkasan akurat dan santai
        system_prompt = (
            "Anda adalah asisten cerdas. Tugas Anda adalah meringkas teks yang diberikan menjadi "
            "maksimal 3-5 poin penting dalam Bahasa Indonesia yang santai tapi profesional. "
            "Langsung berikan ringkasannya, jangan pakai sapaan atau perkenalan diri."
        )

        payload = {
            "model": summary_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.text.strip()}
            ],
            "max_tokens": 1000
        }
        
        response = requests.post(sumopod_url, json=payload, headers=headers, timeout=60)
        
        if response.status_code != 200:
            return JSONResponse(content={"error": "Summary Model Error", "detail": response.text}, status_code=response.status_code)

        result = response.json()
        hasil_ringkasan = result["choices"][0]["message"]["content"].strip()
        
        # Bersihkan jika AI masih bandel ngasih intro
        if "---" in hasil_ringkasan: hasil_ringkasan = hasil_ringkasan.split("---")[0]
        
        print("-> Ringkasan Berhasil!")
        return JSONResponse({"summary": hasil_ringkasan.strip()})
        
    except Exception as e:
        return JSONResponse(content={"error": "Summary System Error", "detail": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"-> Backend Dual-Brain berjalan di port {port}...")
    uvicorn.run("ocr_ai:app", host="0.0.0.0", port=port, reload=True)
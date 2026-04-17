import os
import base64
import requests
import gc
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load variabels dari .env (opsional untuk testing lokal)
load_dotenv()

app = FastAPI(title="OCR Vision Bridge API")

# [CORS POLICY]: Mengizinkan akses Flutter atau klien dari domain apapun ('*')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-ocr")
async def process_ocr(file: UploadFile = File(...)):
    print(f"-> Menerima file: {file.filename}, Tipe: {file.content_type}")
    
    # 1. Validasi MIME Type apakah file benar-benar gambar
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
    
    # 2. Baca binary image file dan encode ke Base64
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
    
    # [MEMORY MANAGEMENT]: Bebaskan memori binary sesegera mungkin
    del contents
    gc.collect()

    # 3. Setup koneksi ke SumoPod API (Utamakan membaca dari os.getenv / konfigurasi Render.com)
    api_key = os.getenv("API_KEY_SUMOPOD") or os.getenv("SUMOPOD_API_KEY", "")
    api_key = api_key.strip()
    
    if not api_key:
        raise HTTPException(status_code=500, detail="Server Error: Kredensial API Key gagal dibaca oleh server web.")

    sumopod_url = os.getenv("SUMOPOD_URL", "https://ai.sumopod.com/v1/chat/completions")
    sumopod_vision_model = os.getenv("SUMOPOD_VISION_MODEL", "gpt-4o-mini")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    censored_key = api_key[:4] + "***" + api_key[-4:] if len(api_key) > 8 else "***"
    print(f"-> Mengirim request ke SumoPod URL: {sumopod_url}")
    print(f"-> Menggunakan model: {sumopod_vision_model}")
    print(f"-> Headers dikirim: {{'Authorization': 'Bearer {censored_key}', 'Content-Type': 'application/json'}}")

    # [CLEAN OUTPUT]: Pesan instruksi OCR yang strict tanpa basa-basi
    prompt_text = (
        "Tugas Anda adalah melakukan OCR. Ekstrak semua teks dari gambar yang diberikan secara akurat. "
        "Jangan tambahkan kata-kata pembuka, jangan tambahkan identitas Anda, jangan memberikan analisis. "
        "HANYA teks yang ada di dalam gambar."
    )

    payload = {
        "model": sumopod_vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{file.content_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2000
    }
    
    # 4. Melakukan HIT API ke SumoPod secara synchronous
    try:
        response = requests.post(sumopod_url, json=payload, headers=headers, timeout=60)
        
        # [MEMORY MANAGEMENT]: Bersihkan Payload base64 teks gambar yang besar setelah request terkirim
        del base64_image
        del payload
        gc.collect()

        print(f"-> Balasan dari SumoPod HTTP Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"-> Detail Error SumoPod: {response.text}")
            return JSONResponse(
                content={"error": f"SumoPod API membalas status {response.status_code}", "detail": response.text}, 
                status_code=response.status_code
            )

        result = response.json()
        raw_text = result["choices"][0]["message"]["content"]
        
        # Eksekusi pembersihan string dari "sampah" wrapper API/Claude
        if "---" in raw_text:
            raw_text = raw_text.split("---")[0]
        if "Namun, saya catatan" in raw_text:
            raw_text = raw_text.split("Namun, saya catatan")[0]
            
        hasil_teks = raw_text.strip()
        print("-> Teks berhasil diekstrak dan dibersihkan!")
        
    except requests.exceptions.RequestException as e:
        print(f"-> Exception (Request Error): {str(e)}")
        return JSONResponse(content={"error": "Gagal terhubung ke AI server.", "detail": str(e)}, status_code=500)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"-> Exception (Internal Error): {str(e)}")
        return JSONResponse(content={"error": "Internal Processing Error.", "detail": str(e)}, status_code=500)
        
    # 5. Return JSON ke Flutter
    return JSONResponse({
        "hasil": hasil_teks
    })

# ==============================================================================
# Endpoint /summarize (DINONAKTIFKAN MENTARA UNTUK MENGHEMAT RESOURCES)
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    # [DEPLOYMENT PORT]: Port dinamis menyesuaikan request server dari Render.com
    port = int(os.environ.get("PORT", 8000))
    # Untuk server production uvicorn reload=True biasane dimatikan tapi ditambahkan secara command line
    uvicorn.run("ocr_ai:app", host="0.0.0.0", port=port)
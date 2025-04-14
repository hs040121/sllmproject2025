from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import torch

app = FastAPI()

# CORS 설정 (모든 요청 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로딩 (Gemma 2B 모델 경로 지정)
model_path = "./gemma-2-2b-it"  # 모델 경로 맞게 설정할 것

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.to("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/generate_text")
async def generate_text(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")

        if not prompt:
            return JSONResponse(content={"error": "prompt 값이 비어 있습니다."}, status_code=400)

        # 텍스트 생성
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=500,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            no_repeat_ngram_size=2
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return JSONResponse(content={"result": generated})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
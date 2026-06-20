# Analog IC Studio — Hugging Face Spaces (Docker SDK)
# Flask + ngspice (apt) + sky130 minimal PDK。不裝 torch (DNN 為延遲 import, 無模型時優雅退回)。
FROM python:3.11-slim

# ngspice (Debian apt 內建) — 提供真實 SPICE 模擬引擎
RUN apt-get update && apt-get install -y --no-install-recommends ngspice \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces 以 uid 1000 執行: 建立 user 並讓 /app 可寫 (app 會寫 run.sp/wave.txt 等暫存)
RUN useradd -m -u 1000 user && mkdir -p /app && chown -R user:user /app
WORKDIR /app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

USER user
ENV PORT=7860
EXPOSE 7860
CMD ["python", "web_app.py"]

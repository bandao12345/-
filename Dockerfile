FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 APP_HOST=0.0.0.0 PORT=5003 INKLAB_DATA_DIR=/data INKLAB_DISABLE_LOCAL_LLM=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home inklab && mkdir -p /data && chown -R inklab:inklab /data
USER inklab
EXPOSE 5003
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,json; d=json.load(urllib.request.urlopen('http://127.0.0.1:5003/health')); assert d['ready']"
CMD ["python", "serve.py"]

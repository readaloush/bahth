# Bahth search service. No PyTorch in the image: ONNX Runtime, tokenizers and numpy only.
#   docker build -t bahth .
#   docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" bahth
FROM python:3.12-slim
WORKDIR /app
COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt
COPY bahth/__init__.py bahth/search.py bahth/
COPY service service
ENV BAHTH_ARTIFACTS=/app/artifacts
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]

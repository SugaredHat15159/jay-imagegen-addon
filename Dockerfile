# JAY Image Generation Addon — CPU sd-turbo. Author: SugaredHat15159.
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY run_imagegen.py .
CMD ["python", "run_imagegen.py"]

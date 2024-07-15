FROM python:3.8

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY saliency_map_load.py .
COPY .env .

CMD ["python", "saliency_map_load.py"]
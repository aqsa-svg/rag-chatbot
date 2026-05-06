FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build index at container build time (requires data/conversations.csv)
RUN python build_index.py

EXPOSE 5000

CMD ["python", "app.py"]

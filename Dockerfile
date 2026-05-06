FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=backend
ENV PORT=7860

RUN python build_index.py

EXPOSE 7860

CMD ["python", "app.py"]
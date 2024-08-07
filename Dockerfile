FROM python:3.11-slim

WORKDIR /consultant

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]

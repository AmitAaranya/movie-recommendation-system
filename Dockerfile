FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py
COPY src /app/src
COPY data /app/data

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]

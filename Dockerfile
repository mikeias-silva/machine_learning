FROM python

WORKDIR /app

COPY requeriments.txt .

RUN pip install --no-cache-dir -r requeriments.txt
FROM python:3.7-slim

WORKDIR /inference

COPY requirements.txt .

RUN pip3 install -Ur requirements.txt --upgrade

COPY . .

ENTRYPOINT ["python3", "inference.py"]

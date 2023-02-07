FROM python:3.9.9-slim-buster
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pyannote.audio

WORKDIR /tmp/requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

ENV INPUT_AUDIO_FILE "<s3_URI>"
ENV OUTPUT_BUCKET "<s3_URI>"
ENV AWS_ACCESS_KEY_ID "<AWS_ACCESS_KEY_ID>"
ENV AWS_SECRET_ACCESS_KEY "<AWS_SECRET_ACCESS_KEY>"
ENV HUGGINGFACE_TOKEN "<HUGGINGFACE_TOKEN>"

WORKDIR /workspace
COPY src/ .
#RUN pip install nemo_toolkit['all']
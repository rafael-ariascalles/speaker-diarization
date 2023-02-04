FROM nvcr.io/nvidia/pytorch:22.12-py3
RUN rm -r ./*
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    libsndfile1 sox \
    libfreetype6 \
    swig \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/NVIDIA/NeMo.git@r1.14.0

WORKDIR /tmp/torchaudio_build
COPY install_torchaudio_latest.sh /tmp/torchaudio_build/scripts/installers/
RUN /bin/bash /tmp/torchaudio_build/scripts/installers/install_torchaudio_latest.sh

WORKDIR /tmp/requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

ENV INPUT_AUDIO_FILE "<s3_URI>"
ENV OUTPUT_BUCKET "<s3_URI>"
ENV AWS_ACCESS_KEY_ID "<AWS_ACCESS_KEY_ID>"
ENV AWS_SECRET_ACCESS_KEY "<AWS_SECRET_ACCESS_KEY>"

WORKDIR /workspace
COPY src/ .
#RUN pip install nemo_toolkit['all']
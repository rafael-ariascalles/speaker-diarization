# speaker-diarization
Approaches for Speaker Diarization


## Enviroment Variables
```.env
AWS_ACCESS_KEY_ID=<>
AWS_SECRET_ACCESS_KEY=<>
INPUT_AUDIO_FILE=s3://stak-diarization-input/audio/project_1663714_vocals.wav
OUTPUT_BUCKET=s3://stak-diarization-output/files
```

## NeMo Diarization
```bash
cd speaker-diarization/NVIDIA-NeMO
sudo docker build . -t nemo
sudo docker run --gpus all --rm --name nm --shm-size=30gb --env-file=.env -it nemo python main.py
```
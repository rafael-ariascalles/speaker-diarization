# speaker-diarization
Approaches for Speaker Diarization


## Enviroment Variables
```.env
AWS_ACCESS_KEY_ID=<>
AWS_SECRET_ACCESS_KEY=<>
INPUT_AUDIO_FILE=<S3://PATH_TO_FILE>
OUTPUT_BUCKET=<S3://PATH_TO_BUCKET>
```

## NeMo Diarization
```bash
cd speaker-diarization/NVIDIA-NeMO
sudo docker build . -t nemo
sudo docker run --gpus all --rm --name nm --shm-size=30gb --env-file=.env -it nemo python main.py
```

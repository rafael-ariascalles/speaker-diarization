# speaker-diarization
Approaches for Speaker Diarization

## NeMo Diarization

```bash
cd speaker-diarization/NVIDIA-NeMO
sudo docker run --gpus all --rm --name nm --shm-size=30gb --env-list=.env nemo
sudo docker run --gpus all --rm --name nm --shm-size=30gb --env-file=.env -it nemo python main.py
```
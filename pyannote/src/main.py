#
import json
import os
#
import IPython
import matplotlib.pyplot as plt
import numpy as np
import librosa
#
import wget
from omegaconf import OmegaConf
import soundfile
#
import json
import io
import boto3
import awswrangler as wr
#
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

ROOT = os.getcwd()
data_dir = os.path.join(ROOT,'data')
os.makedirs(data_dir, exist_ok=True)

output_dir = os.path.join(ROOT, 'outputs')
os.makedirs(output_dir, exist_ok=True)

MODEL_CONFIG = "diar_infer_telephonic.yaml"
pretrained_vad = 'vad_multilingual_marblenet'
pretrained_speaker_model = 'titanet_large'
pretrained_diarizer_model = 'diar_msdd_telephonic'


meta = {
    'audio_filepath': "mono_file.wav", 
    'offset': 0, 
    'duration':None, 
    'label': 'infer', 
    'text': '-', 
    'num_speakers': 2, 
    'rttm_filepath': None, 
    'uem_filepath' : None
}
with open('data/input_manifest.json','w') as fp:
    json.dump(meta,fp)
    fp.write('\n')

if __name__ == "__main__":

    s3_path = os.getenv("INPUT_AUDIO_FILE")
    buffer = io.BytesIO() 
    sess = boto3.Session(aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    wr.s3.download(path=s3_path,local_file=buffer,boto3_session=sess)
    buffer.seek(0)
    
    filename = s3_path.split("/")[-1].split(".")[0]

    speaker_ts = []
    with open(result_rttm, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])


    with open(input_next_step, "w") as fp:
        json.dump(speaker_ts, fp)

    wr.s3.upload(local_file=result_rttm, path="{}/{}.rttm".format(os.getenv("OUTPUT_BUCKET"),filename))    
    wr.s3.upload(local_file=input_next_step, path="{}/{}_speaker.txt".format(os.getenv("OUTPUT_BUCKET"),filename))

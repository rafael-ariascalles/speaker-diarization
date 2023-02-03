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
    'num_speakers': None, 
    'rttm_filepath': None, 
    'uem_filepath' : None
}
with open('data/input_manifest.json','w') as fp:
    json.dump(meta,fp)
    fp.write('\n')

if __name__ == "__main__":


    signal, sample_rate = librosa.load("vocals.wav", sr=None)
    soundfile.write("mono_file.wav", signal, sample_rate, "PCM_24")


    config = OmegaConf.load(MODEL_CONFIG)

    config.diarizer.manifest_filepath = 'data/input_manifest.json'
    config.diarizer.out_dir = output_dir

    config.diarizer.oracle_vad = False # ----> ORACLE VAD 
    config.diarizer.clustering.parameters.oracle_num_speakers = False


    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model # ---------> MODEL
    config.diarizer.msdd_model.model_path = pretrained_diarizer_model # ---------> MODEL
    config.diarizer.vad.model_path = pretrained_vad # ---------> MODEL
    
    #config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.75,1.5,1.0]
    #config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.5,0.25] 
    #config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1]

    config.num_workers = 1 # Hangup if set no another number

    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    
    system_vad_msdd_model = NeuralDiarizer(cfg=config)
    system_vad_msdd_model.diarize()


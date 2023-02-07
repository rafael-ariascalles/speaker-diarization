import json
import os
import numpy as np
import json
import io
import boto3
import awswrangler as wr
from pyannote.audio import Pipeline
import ffmpeg
from pydub import AudioSegment

ROOT = os.getcwd()
data_dir = os.path.join(ROOT,'data')
os.makedirs(data_dir, exist_ok=True)

output_dir = os.path.join(ROOT, 'outputs')
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    #Data Step
    print(".............audio S3.........................")
    s3_path = os.getenv("INPUT_AUDIO_FILE")
    print(s3_path)
    buffer = io.BytesIO() 
    sess = boto3.Session(aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    wr.s3.download(path=s3_path,local_file=buffer,boto3_session=sess)
    buffer.seek(0)

    print(".............transforming audio file...........")
    #ffmpeg.input("pipe:").output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr).run(input=buffer.get_values(),cmd=["ffmpeg", "main.wav"], capture_stdout=True, capture_stderr=True)
    audio_segment = AudioSegment.from_file(buffer)
    audio_segment.export("main.wav",format="wav")
    #Data Folder
    filename = s3_path.split("/")[-1].split(".")[0]
    #Implementation of the model

    print(".............download_model.....................")
    diarization_model = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))

    #rttm result
    print(".............inferences.........................")
    diarization = diarization_model("main.wav")

    print(".............rttm generation.....................")
    result_rttm = "{}.rttm".format(filename)
    with open(result_rttm, "w") as rttm:
        diarization.write_rttm(rttm)
    #speaker information

    print(".............speaker generation...................")
    speaker_ts = []
    for i,(turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        #Millisecond
        start = int(turn.start*1_000)
        end = int(turn.end*1_000)
        speaker_id = int(speaker.split("_")[-1])
        speaker_ts.append([start,end,speaker_id])

    input_next_step = "{}_speaker.txt".format(filename)
    with open(input_next_step, "w") as fp:
        json.dump(speaker_ts, fp)
    
    print(".............uploading.............................")
    wr.s3.upload(local_file=result_rttm, path="{}/{}.rttm".format(os.getenv("OUTPUT_BUCKET"),filename))    
    wr.s3.upload(local_file=input_next_step, path="{}/{}_speaker.txt".format(os.getenv("OUTPUT_BUCKET"),filename))

    print(".............finish................................")

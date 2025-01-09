import subprocess
import os
import json
import numpy as np
from datetime import datetime, timedelta
import torch
import whisper

SAMPLE_RATE = 16000
CHUNK_SIZE = 4000

class Transcriber:
    def __init__(self, model_name="base"):
        # Load Whisper model when initializing
        self.model = whisper.load_model(model_name)

    def fmt(self, data):
        # Process a segment dictionary into the desired format
        start = data.get("start", 0)
        end = data.get("end", 0)
        
        return {
            "start_time": str(timedelta(seconds=start)).split(".")[0].zfill(8), 
            "end_time": str(timedelta(seconds=end)).split(".")[0].zfill(8), 
            "description": data["text"],
            "duration": str(round(end - start)),
        }

    def transcribe(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} was not found.")

        transcription = []
        start_time = datetime.now()


        audio = whisper.load_audio(filename)
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).to(self.model.device)

        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(self.model, mel, options)

        # Format the transcription results
        for item in result['segments']:
            transcription.append(self.fmt(item))

        end_time = datetime.now()
        time_elapsed = end_time - start_time
        print(f"Time elapsed: {time_elapsed}")

        return {
            "segments": transcription,
        }


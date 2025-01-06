import json
import subprocess
import os
import sys
import wave
from scripts.vosk_trascriber import Transcriber
import torch
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def transcribe(input_file, model='large-v3'):
    print(f"Iniciando transcrição de {input_file}...")
    start_time = time.time()  # Tempo de início da transcrição
    
    output_folder = 'tmp'
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    srt_file = os.path.join(output_folder, f"{base_name}.srt")

    # Verifica se o arquivo SRT já existe
    if os.path.exists(srt_file):
        print(f"O arquivo {srt_file} já existe. Pulando a transcrição.")
        return srt_file

    # Verifica se há uma GPU disponível e define o tipo de processamento
    try:
        end_time = time.time()  # Tempo de término da transcrição
        elapsed_time = end_time - start_time  # Tempo total de execução
        
        audio_file = "tmp/input_video.mp4"
        device = "cpu"
        
        model_path = "vosk-model-small-pt-0.3"
        transcriber = Transcriber(model_path)

        result = transcriber.transcribe(audio_file)

        with open(srt_file, "w") as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)

        # Cálculo de minutos e segundos
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        print(f"Transcrição concluída. Saída salva em {srt_file}.")
        print(f"Levou {minutes} minutos e {seconds} segundos para transcrever usando {device}.")  # Mostra o tempo e o dispositivo
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Erro durante a transcrição: {e}")
        print(f"Saída de erro: {e.stderr}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

    # Verifica se o arquivo SRT foi criado
    if os.path.exists(srt_file):
        print(f"Arquivo SRT {srt_file} criado com sucesso.")
    else:
        print("Aviso: O arquivo SRT não foi criado como esperado.")

    return srt_file

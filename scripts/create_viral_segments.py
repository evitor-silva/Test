import json
from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def create(num_segments, viral_mode, themes, tempo_minimo, tempo_maximo):
    # Lê o conteúdo da transcrição do vídeo em formato SRT
    with open('tmp/input_video.srt', 'r') as f:
        content = f.read()

    # Dividir o conteúdo da legenda em chunks de tamanho apropriado
    chunk_size = 17400  # Tamanho de chunk de 14,000 a 17,400 caracteres
    chunks = []
    start = 0

    # Divisão do conteúdo em chunks respeitando quebras de linha
    while start < len(content):
        end = min(start + chunk_size, len(content))
        if end < len(content):
            end = content.rfind('\n', start, end)
            if end == -1:
                end = start + chunk_size
        chunks.append(content[start:end])
        start = end

    # Analisar sentimento e identificar melhores momentos
    best_moments = []
    for chunk in chunks:
        sentiment_score = analyze_sentiment(chunk)
        if sentiment_score > 0.5:  # Ajuste o valor conforme necessário
            best_moments.append({
                "chunk": chunk,
                "sentiment_score": sentiment_score
            })

    # Organizar em 'num_segments' segmentos ou baseados no 'tempo_minimo' e 'tempo_maximo'
    best_moments = sorted(best_moments, key=lambda x: x['sentiment_score'], reverse=True)
    selected_moments = best_moments[:num_segments]

    # Imprimir os melhores momentos
    print(json.dumps(selected_moments, indent=4))


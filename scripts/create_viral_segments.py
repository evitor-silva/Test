import json

def create(num_segments, viral_mode, themes, tempo_minimo, tempo_maximo):
    # Lê o conteúdo da transcrição do vídeo em formato SRT
    with open('tmp/input_video.srt', 'r') as f:
        content = f.read()

    # Sistema de análise de segmentos virais
    system = """You are a Viral Segment Identifier, an AI system that analyzes a video's transcript and predicts which segments might go viral on social media platforms. You use factors such as emotional impact, humor, unexpected content, and relevance to current trends to make your predictions. You return a structured text document detailing the start and end times, the description, the duration, and a viral score for the potential viral segments."""

    # Template JSON para os segmentos virais
    json_template = '''
    {
        "segments": [
            {
                "title": "Suggested Viral Title",
                "start_time": "00:00:00", #HH:MM:SS
                "end_time": "00:00:00", #HH:MM:SS
                "description": "Description of the text",
                "duration": 0,
                "score": 0  # Probability of going viral (0-100)
            }
        ]
    }
    '''

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

    # Definição do tipo de análise (viral ou por temas)
    if viral_mode:
        task_type = f"analyze the segment for potential virality and identify {num_segments} most viral segments from the transcript"
    else:
        task_type = f"analyze the segment for potential virality and identify {num_segments} best segments based on the list of themes: {themes}."

    output_texts = []
    
    # Montar a saída de texto para cada chunk
    for i, chunk in enumerate(chunks):
        if len(chunks) == 1:
            output_text = f"""
    {system}\n
    Given the following video transcript, {task_type}. Each segment must have a duration between {tempo_minimo} and {tempo_maximo} seconds. It is MANDATORY to respect the specified number of viral segments, the minimum duration, and the maximum duration. Additionally, the cuts MUST MAKE SENSE and cannot end abruptly without context. The provided transcript is as follows:
    {chunk}
    Based on your analysis, return a structured text document containing the timestamps (start and end), the description of the viral part, its duration, a suggested viral title, and a score indicating the probability of going viral. Please follow this format for each segment.
    {json_template}
    The total duration must be within a minimum time of {tempo_minimo} seconds and a maximum time of {tempo_maximo} seconds.
    """
        else:
            if i == 0:
                output_text = f"""
    {chunk}
    """
            elif i < len(chunks) - 1:
                output_text = f"""
    Vou enviar outra parte da legenda. Analise e responda com OK para enviar a próxima parte.

    {chunk}
    """
            else:
                output_text = f"""
    {chunk}\n\n
    {system}\n
    Given the following video transcript, {task_type}. Each segment must have a duration between {tempo_minimo} and {tempo_maximo} seconds. It is MANDATORY to respect the specified number of viral segments, the minimum duration, and the maximum duration. Additionally, the cuts MUST MAKE SENSE and cannot end abruptly without context. The provided transcript is as follows:
    {chunk}
    Based on your analysis, return a structured text document containing the timestamps (start and end), the description of the viral part, its duration, a suggested viral title, and a score indicating the probability of going viral. Please follow this format for each segment. Leave the 'title' and 'description' in the language of the subtitles.
    {json_template}
    The total duration must be within a minimum time of {tempo_minimo} seconds and a maximum time of {tempo_maximo} seconds.
    """

        output_texts.append(output_text)

    # Exibe os textos gerados (pode ser ajustado para salvar ou processar)
    for text in output_texts:
        print(text)
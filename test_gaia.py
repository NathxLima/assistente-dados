from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests

# Nome correto do modelo GAIA
MODEL_NAME = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"

# Baixa o processor (tokenizer + feature extractor)
print("🔹 Carregando processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# Baixa o modelo
print("🔹 Carregando modelo...")
model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME)

# Prompt de texto
prompt = "Explique em português simples o que é aprendizado de máquina."

# Caso queira usar apenas texto (sem imagem), use None na imagem:
image = None

# (Opcional) Se quiser testar imagem + texto, descomente abaixo:
# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
# image = Image.open(requests.get(url, stream=True).raw)
# prompt = "O que aparece nesta imagem?"

# Prepara inputs
inputs = processor(
    images=image,
    text=prompt,
    return_tensors="pt"
)

# Gera resposta com parâmetros customizados
print("🔹 Gerando resposta...")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

# Decodifica a saída
response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

# Exibe resultado
print("\n🔹 Resposta gerada:\n")
print(response)

import json
import openai
import os

from dotenv import load_dotenv

ARQUIVO_JSONL = "data/sebrae_finetune.jsonl"
ARQUIVO_SAIDA = "data/sebrae_finetune_corrigido.jsonl"

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

with open(ARQUIVO_JSONL, "r", encoding="utf-8") as entrada, open(ARQUIVO_SAIDA, "w", encoding="utf-8") as saida:
    for linha in entrada:
        exemplo = json.loads(linha)

        mensagens = exemplo["messages"]

        resposta = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=mensagens,
            temperature=exemplo.get("temperature", 0.7)
        )

        mensagens.append({
            "role": "assistant",
            "content": resposta.choices[0].message.content.strip()
        })

        json.dump({"messages": mensagens, "temperature": exemplo.get("temperature", 0.7)}, saida, ensure_ascii=False)
        saida.write("\n")

print("✅ Respostas preenchidas com sucesso.")





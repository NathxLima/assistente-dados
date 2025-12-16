import openai
import os
from dotenv import load_dotenv

# Carrega a variável OPENAI_API_KEY do .env
load_dotenv()

# Cria cliente com a chave da API
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Inicia o fine-tuning
response = client.fine_tuning.jobs.create(
    training_file="file-2mCeCeqi2zbxnP6uQybXre",  # ID do seu arquivo JSONL
    model="gpt-3.5-turbo"
)

print("Fine-tuning iniciado com sucesso!")
print("Job ID:", response.id)



import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Inicia o fine-tuning
response = client.fine_tuning.jobs.create(
    training_file="file-PGMTiP9aCK45agNQrSAPd9",
    model="gpt-3.5-turbo"
)

print("Fine-tuning iniciado com sucesso!")
print("Job ID:", response.id)
# ingest_mysql.py
import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Carregar variáveis do .env
load_dotenv()

host = os.getenv("MYSQL_HOST")
port = os.getenv("MYSQL_PORT")
user = os.getenv("MYSQL_USER")
password = os.getenv("MYSQL_PASSWORD")
database = os.getenv("MYSQL_DATABASE")
table = os.getenv("MYSQL_TABLE")

# Construir URI de conexão
uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
engine = create_engine(uri)

# Ler dados do MySQL
query = f"SELECT * FROM {table}"
df = pd.read_sql(query, engine)

# Transformar cada linha em texto descritivo
def formatar_linha(linha):
    return (
        f"O aluno {linha['nome']}, de gênero {linha['genero']}, "
        f"estuda o curso de {linha['curso']} na modalidade {linha['modalidade']}. "
        f"Sua média final é {linha['media_final']} com {linha['reprovacoes']} reprovações. "
        f"Frequência: {linha['faltas_pct']}% de faltas. "
        f"Recebe bolsa: {linha['bolsa']}. Trabalha: {linha['trabalha']}. "
        f"Tem renda familiar: {linha['renda_familiar']}. Evadiu: {linha['evadiu']}."
    )

documentos_texto = [formatar_linha(row) for _, row in df.iterrows()]

# Vetorização
print("🔍 Carregando embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

persist_path = f"data/mysql_{database.lower()}"
os.makedirs(persist_path, exist_ok=True)

print("💾 Armazenando vetores em:", persist_path)
Chroma.from_texts(documentos_texto, embedding=embeddings, persist_directory=persist_path)

print("✅ Vetorização finalizada com sucesso!")

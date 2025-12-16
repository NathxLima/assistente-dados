import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

pasta = os.path.join("data", "machine_learning")
db = Chroma(persist_directory=pasta, embedding_function=embeddings)

print("Pasta:", pasta)
print("Qtd de vetores:", db._collection.count())

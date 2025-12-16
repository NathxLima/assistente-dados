import os
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CAMINHO_DOCS = "docs"
CAMINHO_DATA = "data"
BATCH_SIZE = 1000

def texto_valido(doc) -> bool:
    try:
        texto = doc.page_content
        return (
            isinstance(texto, str)
            and texto.strip() != ""
            and "\x00" not in texto
            and not texto.isspace()
        )
    except Exception:
        return False

def carregar_pdfs_da_pasta(pasta_docs: str):
    documentos = []
    total_arquivos = 0
    erros = []

    for root, _, files in os.walk(pasta_docs):
        for arquivo in files:
            if not arquivo.lower().endswith(".pdf"):
                continue
            total_arquivos += 1
            caminho_arquivo = os.path.join(root, arquivo)
            print(f"📄 Lendo: {caminho_arquivo}")
            try:
                loader = PyPDFLoader(caminho_arquivo, extract_images=False, extraction_mode="plain")
                docs = loader.load()

                # (Opcional, mas recomendado) guardar fonte
                for d in docs:
                    d.metadata["source"] = os.path.basename(caminho_arquivo)

                documentos.extend(docs)
                print(f"✅ OK: {caminho_arquivo}")
            except Exception as e:
                print(f"❌ Erro ao ler {caminho_arquivo}: {e}")
                erros.append((caminho_arquivo, str(e)))

    print(f"🔎 PDFs encontrados: {total_arquivos} | Erros: {len(erros)}")
    return documentos

def vetorizar_tema(nome_tema: str, pasta_docs_tema: str, pasta_data_tema: str, embeddings):
    print(f"\n==============================")
    print(f"🚀 Tema: {nome_tema}")
    print(f"📁 Docs: {pasta_docs_tema}")
    print(f"🧠 Data: {pasta_data_tema}")
    print(f"==============================")

    docs = carregar_pdfs_da_pasta(pasta_docs_tema)
    print(f"📚 Documentos carregados: {len(docs)}")

    if not docs:
        print("⚠️ Nada para vetorizar neste tema. Pulando.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs_split = splitter.split_documents(docs)
    docs_filtrados = [d for d in docs_split if texto_valido(d)]

    print(f"✂️ Chunks criados: {len(docs_split)} | ✅ Válidos: {len(docs_filtrados)}")
    if not docs_filtrados:
        print("⚠️ Nenhum chunk válido. Pulando.")
        return

    os.makedirs(pasta_data_tema, exist_ok=True)

    chroma_db = None
    for i in tqdm(range(0, len(docs_filtrados), BATCH_SIZE), desc=f"🧠 Vetorizando {nome_tema}"):
        batch_docs = docs_filtrados[i:i + BATCH_SIZE]
        batch_docs = [d for d in batch_docs if texto_valido(d)]
        if not batch_docs:
            continue

        try:
            if chroma_db is None:
                chroma_db = Chroma.from_documents(
                    documents=batch_docs,
                    embedding=embeddings,
                    persist_directory=pasta_data_tema
                )
            else:
                chroma_db.add_documents(batch_docs)
        except Exception as e:
            print(f"❌ Erro batch {i}-{i + BATCH_SIZE}: {e}")

    if chroma_db:
        chroma_db.persist()
        print(f"✅ Base vetorial do tema '{nome_tema}' persistida com sucesso!")
    else:
        print(f"⚠️ Nada foi persistido para o tema '{nome_tema}'.")

def main():
    print("🚀 Iniciando ingestão por tema...")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Cada subpasta dentro de /docs vira um tema
    subpastas = [
        d for d in os.listdir(CAMINHO_DOCS)
        if os.path.isdir(os.path.join(CAMINHO_DOCS, d))
    ]

    if not subpastas:
        print("⚠️ Nenhuma subpasta de tema encontrada em /docs.")
        return

    for tema in subpastas:
        pasta_docs_tema = os.path.join(CAMINHO_DOCS, tema)
        pasta_data_tema = os.path.join(CAMINHO_DATA, tema)
        vetorizar_tema(tema, pasta_docs_tema, pasta_data_tema, embeddings)

    print("\n🏁 Finalizado.")

if __name__ == "__main__":
    main()





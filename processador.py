# ================== IMPORTS ==================
import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ================== CONFIG ==================
load_dotenv()
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(
    page_title="🤖 Nathal.IA",
    layout="wide"
)

CAMINHO_VETORIAL = "data/global"
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ================== VETORES ==================
@st.cache_resource
def carregar_vetores():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory=CAMINHO_VETORIAL,
        embedding_function=embeddings
    )

try:
    vetores = carregar_vetores()
    retriever = vetores.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
except Exception as e:
    st.error(f"Erro ao carregar base vetorial: {e}")
    st.stop()

# ================== LLM ==================
try:
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.2,
        max_tokens=900,
        api_key=OPENAI_API_KEY
    )
except Exception as e:
    st.error(f"Erro ao iniciar modelo OpenAI: {e}")
    st.stop()

# ================== QA CHAIN ==================
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_pt}
)

# ================== FUNÇÃO HF ==================
def busca_semantica_mcp(consulta, tipo="spaces"):
    if not HF_TOKEN:
        return []

    url = f"https://huggingface.co/mcp/search/{tipo}?q={consulta}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 200:
        return response.json()[:3]
    return []

# ================== UI ==================
st.markdown("""
<div class="header-brand">
    <span style='font-size: 48px;'>🎲</span>
    <div>
        <h1>Nathal.IA</h1>
        <p>Sua assistente de dados personalizada</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ================== HISTÓRICO ==================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def mostrar_historico():
    for item in st.session_state.chat_history:
        st.markdown(
            f'<div class="bubble user-msg">🧠 Você: {item["pergunta"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="bubble bot-msg">🤖 Resposta: {item["resposta"]}</div>',
            unsafe_allow_html=True
        )
        with st.expander("📌 Fontes utilizadas"):
            for i, doc in enumerate(item["fontes"]):
                st.markdown(f"**Documento {i+1}:**")
                st.code(doc.page_content[:500] + "...")

mostrar_historico()

# ================== INPUT ==================
with st.form("pergunta_form", clear_on_submit=True):
    pergunta = st.text_input("Digite sua pergunta...")
    enviar = st.form_submit_button("Enviar")

# ================== PROCESSAMENTO ==================
if enviar and pergunta:
    with st.spinner("🤖 Nathal.IA está pensando..."):
        try:
            resposta = qa_chain.invoke({"question": pergunta})
            texto = resposta.get("answer", "").strip().lower()

            resposta_generica = any(p in texto for p in [
                "você pode procurar",
                "uma possibilidade seria",
                "você também pode",
                "a plataforma hugging face"
            ])

            if not texto or resposta_generica:
                sugestoes = busca_semantica_mcp(pergunta)
                resposta_final = "🔍 Referências externas encontradas:\n\n"
                for item in sugestoes:
                    resposta_final += f"• {item.get('title')} — {item.get('url')}\n"
            else:
                resposta_final = resposta["answer"]

            st.session_state.chat_history.append({
                "pergunta": pergunta,
                "resposta": resposta_final,
                "fontes": resposta.get("source_documents", [])
            })

            st.rerun()

        except Exception as e:
            st.error(f"Erro ao gerar resposta: {e}")

# ================== RODAPÉ ==================
st.markdown("""
<hr>
<p style="text-align:center; color:#888; font-size:14px;">
Desenvolvido por <strong>Nathália Lima</strong>
</p>
""", unsafe_allow_html=True)

# -*- coding: utf-8 -*-

import os
import streamlit as st
from dotenv import load_dotenv
import requests
load_dotenv()

# Corrige watcher do Streamlit
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
st.set_page_config(page_title="🎲 Nathal.IA", layout="wide")

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Autenticação personalizada
USUARIOS_AUTORIZADOS = {
    "Carlos Cardoso": "senha01",
    "Sergio Wechsler": "senha02",
    "Ricardo Frugoni": "senha03",
    "Cintia Senem": "senha04",
    "Hector Giacon": "senha05",
    "Hélio Ribeiro": "senha06",
    "Flavio Amadio": "senha07",
    "Lisan Durão": "senha08",
    "Nayara Ferreira": "senha09",
    "Brendon Alcantara": "senha10",
    "admin": "senha11"
}

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.title("🔐 Acesso Restrito — Nathal.IA")
    with st.form("login_form"):
        usuario = st.text_input("Usuário")
        senha = st.text_input("Senha", type="password")
        entrar = st.form_submit_button("Entrar")

    if entrar:
        if usuario in USUARIOS_AUTORIZADOS and USUARIOS_AUTORIZADOS[usuario] == senha:
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.error("Usuário ou senha inválidos.")
    st.stop()

# Caminho base dos documentos
CAMINHO_DOCS = "docs"

# Função para identificar o tema dominante da pergunta
def identificar_tema(pergunta):
    pergunta = pergunta.lower()
    if any(p in pergunta for p in ["estatística", "probabilidade", "média", "variância"]):
        return "estatistica_basica"
    elif any(p in pergunta for p in ["crédito", "inadimplência", "cooperativa", "risco"]):
        return "financas_credito"
    elif any(p in pergunta for p in ["inteligência artificial", "ia", "agente"]):
        return "inteligencia_artificial"
    elif any(p in pergunta for p in ["machine learning", "modelo preditivo", "regressão logística"]):
        return "machine_learning"
    elif any(p in pergunta for p in ["negócio", "empresa", "churn"]):
        return "negocios_geral"
    elif any(p in pergunta for p in ["sql", "select", "join"]):
        return "SQL"
    elif any(p in pergunta for p in ["python", "pandas", "automação"]):
        return "programacao_python"
    else:
        return "outros"

@st.cache_resource
def carregar_vetores(pasta_principal):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    caminhos = [os.path.join(CAMINHO_DOCS, pasta_principal)]
    for pasta in os.listdir(CAMINHO_DOCS):
        caminho = os.path.join(CAMINHO_DOCS, pasta)
        if pasta != pasta_principal and os.path.isdir(caminho):
            caminhos.append(caminho)
    vectorstore = Chroma(persist_directory=os.path.join("data", pasta_principal), embedding_function=embeddings)
    return vectorstore

HF_TOKEN = os.getenv("HF_TOKEN")

st.markdown("""
<div class="header-brand" style="display: flex; align-items: center; gap: 16px;">
    <span style='font-size: 72px;'>🎲</span>
    <div>
        <h1 id="nathalia" style="margin: 0;">Nathal.IA</h1>
        <p>Da engenharia à ciência de dados: sua parceira estratégica em IA</p>
    </div>
</div>
<style>
.bubble.user-msg {
    background-color: #2c2c2c;
    font-weight: 600;
    color: #fff;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 16px;
    border-left: 4px solid #3B82F6;
}
.bubble.bot-msg {
    background-color: #1f1f1f;
    color: #eee;
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 32px;
    border-left: 4px solid #10B981;
}
</style>
<div class="chat-container">
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def mostrar_historico():
    for msg in st.session_state.chat_history:
        st.markdown(f'<div class="bubble user-msg">🧠 <strong>Você:</strong> {msg["pergunta"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble bot-msg">🤖 <strong>Resposta:</strong> {msg["resposta"]}</div>', unsafe_allow_html=True)
        with st.expander("📌 Fontes utilizadas"):
            for i, doc in enumerate(msg["fontes"]):
                st.markdown(f"**Documento {i+1}:**")
                st.code(doc.page_content[:500] + "...")

mostrar_historico()

st.markdown("</div>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    with st.form("pergunta_form", clear_on_submit=True):
        nova_pergunta = st.text_input("Digite sua pergunta...", key="input_field")
        enviar = st.form_submit_button("Enviar")
    st.markdown('</div>', unsafe_allow_html=True)

if enviar and nova_pergunta:
    with st.spinner("🤖 Nathal.IA está pensando..."):
        try:
            tema = identificar_tema(nova_pergunta)
            vetores = carregar_vetores(tema)
            retriever = vetores.as_retriever()
            historico_formatado = "\n\n".join([f"Usuário: {h['pergunta']}\nIA: {h['resposta']}" for h in st.session_state.chat_history[-5:]])

            prompt_pt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                Você é a Nathal.IA — uma inteligência artificial especializada em dados, focada em resolver problemas reais de negócio com profundidade técnica. Sua missão é orientar o usuário com clareza, estratégia e precisão — como uma mentora de dados experiente, mas acessível.

                Evite saudações como \"Olá\" ou \"Claro\". Prefira respostas diretas e fluídas, com explicações que combinem teoria e prática. Use linguagem natural, mas sem simplificar em excesso. Quando for possível, forneça recomendações, exemplos aplicados, boas práticas ou sugestões de frameworks e técnicas reais.

                Sempre que o usuário pedir uma explicação, ensine com passos, lógica e decisões práticas — especialmente para temas como recuperação de crédito, segmentação de clientes, modelos preditivos, estatística e machine learning.

                No final da resposta, sugira 2 ou 3 próximos passos relevantes como: \"Se quiser seguir explorando...\", \"Você pode testar...\", \"Uma possibilidade interessante seria...\"

                Histórico:
                {context}

                [FIM DO HISTÓRICO]
                Pergunta: {question}

                Resposta:
                """
            )

            chain = load_qa_chain(
                ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY")),
                chain_type="stuff",
                prompt=prompt_pt
            )

            docs = retriever.get_relevant_documents(nova_pergunta)
            resposta = chain.invoke({
                "input_documents": docs,
                "question": nova_pergunta,
                "context": historico_formatado
            })

            st.session_state.chat_history.append({
                "pergunta": nova_pergunta,
                "resposta": resposta["output_text"].strip(),
                "fontes": docs
            })
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao gerar resposta: {e}")

st.markdown("""
<hr style="margin-top: 80px; border: none; border-top: 1px solid #444;">
<p style="text-align: center; color: #888; font-size: 14px;">
    Desenvolvido por: <strong>Nathália Lima</strong>
</p>
""", unsafe_allow_html=True)

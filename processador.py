import os
import streamlit as st
from dotenv import load_dotenv
import requests
import json
load_dotenv()

# Corrige watcher do Streamlit
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
st.set_page_config(page_title="🤖 Nathal.IA", layout="wide")

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Caminho para base vetorial
CAMINHO_VETORIAL = "data"

@st.cache_resource
def carregar_vetores():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CAMINHO_VETORIAL, embedding_function=embeddings)
    return vectorstore

try:
    vetores = carregar_vetores()
    retriever = vetores.as_retriever()
except Exception as e:
    st.error(f"Erro ao carregar base vetorial: {e}")
    st.stop()

try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
except Exception as e:
    st.error(f"Erro ao iniciar modelo OpenAI: {e}")
    st.stop()

# Prompt ajustado com contexto e sugestões finais
prompt_pt = PromptTemplate.from_template(
    """
    Você é a Nathal.IA — uma inteligência artificial especializada em dados, gentil, comunicativa e inteligente. Você é uma tutora de dados inteligente, paciente e empática. Sua missão é ajudar o usuário a entender conceitos de dados de forma clara, acessível e inspiradora — como se estivesse guiando um colega que quer aprender, e não apenas entregando respostas prontas.

    Responda sempre em português, de forma natural e com transições suaves. Use uma linguagem levemente descontraída, sem parecer robótica, e evite respostas secas ou excessivamente técnicas. Evite começar com saudações como "Olá" ou "Oi", especialmente no meio de uma conversa. Em vez disso, vá direto ao ponto com uma transição fluida. Sinta-se à vontade para usar analogias simples quando for útil. Encoraje o usuário a continuar explorando, se apropriado.

    Baseie-se exclusivamente nos documentos fornecidos.

    No final da resposta, sempre sugira 2 ou 3 possíveis próximos passos úteis relacionados ao tema. Use linguagem como: "Se quiser seguir explorando..." ou "Você também pode..." ou "Outra possibilidade seria..."

    Contexto:
    {context}

    Pergunta: {question}

    Resposta:
    """
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_pt}
)

# Função adicional: Busca semântica com HF MCP
HF_TOKEN = os.getenv("HF_TOKEN")

def busca_semantica_mcp(consulta, tipo="spaces"):
    url = f"https://huggingface.co/mcp/search/{tipo}?q={consulta}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        resultados = response.json()
        return resultados[:3]
    else:
        return []

# Título visual com emoji 🎲 e nome Nathal.IA
st.markdown("""
<div class="header-brand">
    <span style='font-size: 48px;'>🎲</span>
    <div>
        <h1 id="nathalia">Nathal.IA</h1>
        <p>Sua assistente de dados personalizada</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Estilo e layout
st.markdown("""
<style>
body, .main {
    background-color: #121212 !important;
    color: #E0E0E0 !important;
    font-family: 'Segoe UI', sans-serif;
}
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 20px 40px 100px 40px;
    max-width: 900px;
    margin: auto;
    position: relative;
}
h1#nathalia {
    font-size: 42px;
    color: white;
    margin-bottom: 0;
    padding-top: 10px;
}
.header-brand {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 40px 0 40px;
}
.header-brand h1 {
    margin: 0;
    font-size: 36px;
    color: white;
}
.header-brand p {
    margin: 0;
    font-size: 14px;
    color: #bbb;
}
.bubble {
    padding: 14px 20px;
    border-radius: 18px;
    max-width: 90%;
    font-size: 16px;
    line-height: 1.6;
    word-wrap: break-word;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    animation: fadeIn 0.3s ease-in;
    transition: all 0.3s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.user-msg {
    align-self: flex-end;
    background-color: #3B82F6;
    color: white;
    text-align: left;
}
.bot-msg {
    align-self: flex-start;
    background-color: #2C2C2C;
    color: white;
    text-align: left;
}
.chat-input-container {
    position: fixed;
    bottom: 0;
    width: 100%;
    background-color: #121212;
    padding: 20px 40px;
    border-top: 1px solid #444;
    z-index: 100;
}
.stTextInput>div>input {
    background-color: #2a2a2a;
    color: white;
    border-radius: 8px;
    padding: 12px;
    font-size: 16px;
    border: 1px solid #3B82F6;
}
.stButton>button {
    background-color: #3B82F6;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
}
.stButton>button:hover {
    background-color: #2563EB;
}
</style>
<div class="chat-container">
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def mostrar_historico():
    for item in st.session_state.chat_history:
        st.markdown(f'<div class="bubble user-msg">🧠 Você: {item["pergunta"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble bot-msg">🤖 Resposta: {item["resposta"]}</div>', unsafe_allow_html=True)
        with st.expander("📌 Fontes utilizadas"):
            for i, doc in enumerate(item["fontes"]):
                st.markdown(f"**Documento {i+1}:**")
                st.code(doc.page_content[:500] + "...")

mostrar_historico()

st.markdown("</div>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    with st.form("pergunta_form", clear_on_submit=True):
        pergunta = st.text_input("Digite sua pergunta...", key="input_field")
        enviar = st.form_submit_button("Enviar")
    st.markdown('</div>', unsafe_allow_html=True)

if enviar and pergunta:
    with st.spinner("🤖 Nathal.IA está pensando..."):
        try:
            resposta = qa_chain.invoke({"query": pergunta})
            resposta_texto = resposta["result"].strip().lower()
            resposta_muito_genérica = any(
                palavra in resposta_texto for palavra in [
                    "você pode procurar", "uma possibilidade seria",
                    "você também pode", "a plataforma hugging face é uma ótima fonte"
                ]
            )
            if not resposta["result"] or resposta_muito_genérica:
                sugestoes = busca_semantica_mcp(pergunta)
                resultado_extra = "\n\n🔍 Também encontrei essas referências na Hugging Face:\n"
                for item in sugestoes:
                    resultado_extra += f"• {item.get('title', 'Sem título')} - {item.get('url', '')}\n"
                resposta_final = resultado_extra
            else:
                resposta_final = resposta["result"]
            st.session_state.chat_history.append({
                "pergunta": pergunta,
                "resposta": resposta_final,
                "fontes": resposta.get("source_documents", [])
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


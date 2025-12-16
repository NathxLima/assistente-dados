# -*- coding: utf-8 -*-
import os
import json
import time
from pathlib import Path

import bcrypt
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage


# ================== CONFIG INICIAL ==================
load_dotenv()
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
st.set_page_config(page_title="🎲 Nathal.IA", layout="wide")
# ====================================================

# ================== AUTH (login/senha) ==================
@st.cache_data(show_spinner=False, ttl=30)
def carregar_usuarios_hash(auth_users_file: str) -> dict:
    if not auth_users_file:
        raise ValueError("AUTH_USERS_FILE vazio")

    arquivo = Path(auth_users_file)
    if not arquivo.exists():
        raise FileNotFoundError(f"Arquivo de usuários não encontrado: {arquivo}")

    return json.loads(arquivo.read_text(encoding="utf-8"))


def validar_login(usuario: str, senha: str, usuarios_hash: dict) -> bool:
    usuario = (usuario or "").strip()
    if not usuario:
        return False

    hash_str = usuarios_hash.get(usuario)
    if not hash_str:
        return False

    try:
        return bcrypt.checkpw(senha.encode("utf-8"), hash_str.encode("utf-8"))
    except Exception:
        return False


def gate_autenticacao():
    # estado inicial
    st.session_state.setdefault("autenticado", False)
    st.session_state.setdefault("usuario", "")
    st.session_state.setdefault("tentativas", 0)
    st.session_state.setdefault("bloqueado_ate", 0.0)

    # já autenticado → segue o app
    if st.session_state["autenticado"]:
        return

    agora = time.time()
    if agora < st.session_state["bloqueado_ate"]:
        st.title("🔐 Acesso Restrito — Nathal.IA")
        st.warning("Muitas tentativas. Aguarde alguns segundos.")
        st.stop()

    # carrega usuários
    auth_users_file = os.getenv("AUTH_USERS_FILE", "").strip()
    try:
        usuarios_hash = carregar_usuarios_hash(auth_users_file)
    except Exception as e:
        st.error(f"Erro de autenticação: {e}")
        st.stop()

    # tela de login
    st.title("🔐 Acesso Restrito — Nathal.IA")
    with st.form("login_form"):
        usuario = st.text_input("Usuário")
        senha = st.text_input("Senha", type="password")
        entrar = st.form_submit_button("Entrar")

    if entrar:
        if validar_login(usuario, senha, usuarios_hash):
            st.session_state["autenticado"] = True
            st.session_state["usuario"] = usuario.strip()
            st.session_state["tentativas"] = 0
            st.session_state["bloqueado_ate"] = 0.0
            st.rerun()
        else:
            st.session_state["tentativas"] += 1
            st.error("Usuário ou senha inválidos.")

            if st.session_state["tentativas"] >= 5:
                st.session_state["bloqueado_ate"] = time.time() + 20
                st.session_state["tentativas"] = 0

    st.stop()


# ✅ CHAMAR A AUTENTICAÇÃO AQUI (logo após definir)
gate_autenticacao()


def botao_logout():
    if st.session_state.get("autenticado"):
        if st.button("Sair"):
            st.session_state["autenticado"] = False
            st.session_state["usuario"] = ""
            st.rerun()


# ================== SELEÇÃO DE TEMA ==================
def identificar_tema(pergunta):
    TEMAS_DISPONIVEIS = [
        "machine_learning",
        "estatistica_basica",
        "inteligencia_artificial",
        "SQL",
        "programacao_python",
        "financas_credito",
        "negocios_geral",
        "mysql_escola",
        "global",
    ]

    pergunta = (pergunta or "").strip()
    if not pergunta:
        return "global"

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    melhor_tema = "global"
    melhor_score = float("inf")

    for tema in TEMAS_DISPONIVEIS:
        pasta = os.path.join("data", tema)
        if not os.path.exists(pasta):
            continue

        try:
            db = Chroma(persist_directory=pasta, embedding_function=embeddings)
            resultados = db.similarity_search_with_score(pergunta, k=1)

            if not resultados:
                continue

            _, score = resultados[0]
            if score < melhor_score:
                melhor_score = score
                melhor_tema = tema

        except Exception:
            continue

    return melhor_tema


# ================== UI ==================
st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
.chat-container { display: flex; flex-direction: column; margin-bottom: 80px; }
.bubble.user-msg {
    background-color: #343541; color: #fff; padding: 12px 16px;
    border-radius: 12px; margin: 8px 0; align-self: flex-end; max-width: 85%;
}
.bubble.bot-msg {
    background-color: #444654; color: #eee; padding: 12px 16px;
    border-radius: 12px; margin: 8px 0; align-self: flex-start; max-width: 85%;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎲 Nathal.IA")
st.subheader("Da engenharia à ciência de dados: sua parceira estratégica em IA")

# (Opcional) botão logout no topo
with st.sidebar:
    st.write(f"👤 Usuário: **{st.session_state.get('usuario', '')}**")
    botao_logout()

# memória do chat
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )

def mostrar_historico():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            st.markdown(
                f'<div class="bubble user-msg">🧠 Você: {msg.content}</div>',
                unsafe_allow_html=True,
            )
        elif isinstance(msg, AIMessage):
            st.markdown(
                f'<div class="bubble bot-msg">🤖 Resposta: {msg.content}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

mostrar_historico()

with st.form("pergunta_form", clear_on_submit=True):
    nova_pergunta = st.text_input("Digite sua pergunta...", placeholder="O que você quer saber?")
    enviar = st.form_submit_button("Enviar")

if enviar and nova_pergunta:
    with st.spinner("🤖 Nathal.IA está pensando..."):
        try:
            # Identificar o tema
            tema = identificar_tema(nova_pergunta)

            # Caminho da base vetorial do tema
            pasta_vetorial = os.path.join("data", tema)
            if not os.path.exists(pasta_vetorial):
                # fallback seguro
                pasta_vetorial = os.path.join("data", "global")

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vetores = Chroma(persist_directory=pasta_vetorial, embedding_function=embeddings)

            # Retriever (k=4)
            retriever = vetores.as_retriever(search_kwargs={"k": 4})

            # LLM (GPT-4.1-mini)
            llm = ChatOpenAI(
                model="gpt-4.1-mini",
                temperature=0.15,
                max_tokens=900,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            # Prompt
            prompt = PromptTemplate(
                input_variables=["chat_history", "context", "question"],
                template="""
Você é a Nathal.IA — uma assistente estratégica de dados criada por Nathália Lima.

Seu papel é apoiar decisões reais de negócio usando dados, estatística e machine learning.
Você responde como uma cientista de dados experiente, segura e prática, com visão de negócio.

Princípios obrigatórios:
- Responda sempre em português.
- Priorize clareza, direcionamento e impacto no negócio.
- Demonstre domínio técnico, explicando conceitos quando isso ajudar a tomar uma decisão melhor.
- Evite tom acadêmico ou excessivamente professoral.
- Não ensine “por ensinar”: toda explicação deve justificar uma escolha, um risco ou uma priorização.
- Só apresente múltiplos caminhos quando houver uma decisão real a ser feita.
- Nunca crie caminhos artificiais apenas para preencher resposta.

Estrutura esperada da resposta:
1) Contextualize rapidamente o problema de negócio.
2) Explique os conceitos técnicos necessários para embasar a decisão (sem excesso).
3) Organize as opções relevantes, destacando trade-offs reais.
4) Finalize com uma recomendação clara, prática e acionável.

Quando fizer sentido:
- Mostre trade-offs (vantagens, riscos, custos de erro).
- Relacione com métricas, orçamento, capacidade operacional ou impacto financeiro.
- Utilize exemplos aplicáveis a contextos reais (crédito, cobrança, churn, operações, dados).

Uso de documentos (RAG):
- Utilize os documentos fornecidos como base factual.
- Se não houver evidência nos documentos, deixe isso explícito.
- Não extrapole além do que os documentos sustentam.

Fontes específicas:
- Se o usuário mencionar explicitamente um autor, livro ou obra:
  - Utilize EXCLUSIVAMENTE os documentos dessa fonte.
  - Se nenhum trecho dessa fonte estiver presente no contexto recuperado,
    informe claramente que não há evidência suficiente para responder.
  - Nunca utilize outras fontes como substituição.

Regra crítica de uso de exemplos:
- Exemplos, números ou modelos mencionados nos documentos são ilustrativos,
  a menos que o usuário forneça explicitamente dados do seu próprio problema.
- Nunca trate exemplos didáticos dos livros como resultados reais aplicáveis.
- Nunca nomeie modelos como “A” ou “B” se eles não existirem explicitamente no problema do usuário.
- Se o documento trouxer apenas exemplos conceituais, deixe isso claro na resposta.

Postura profissional obrigatória:
- Responda como alguém que será cobrado pelo resultado da decisão.
- Evite respostas neutras ou excessivamente abrangentes.
- Sempre deixe claro:
  • O que eu faria
  • O que eu NÃO faria
  • Por quê
- Se houver incerteza, explicite o risco e proponha mitigação.
- Não liste possibilidades sem hierarquizá-las.

Regra de senioridade:
- Engenharia de Dados → foque em arquitetura, ordem de execução e falhas comuns.
- Análise de Dados → foque em interpretação, priorização e comunicação.
- Ciência de Dados → foque em custo de erro, métricas certas e impacto operacional.
- Nunca misture papéis sem justificativa explícita.

Geração de código:
- Gere código somente quando isso ajudar a implementar, validar ou operacionalizar a decisão.
- Antes de apresentar código, explique brevemente POR QUE essa abordagem técnica é adequada ao contexto.
- O código deve ser funcional, organizado e alinhado ao ambiente mencionado pelo usuário.
- Nunca gere código genérico sem conexão clara com o problema de negócio descrito.

Regra de saída (código):
- Se o usuário pedir explicitamente por código (ex: “monte um código”, “me dê um script”, “quero um exemplo”), forneça código completo e executável.
- Se o usuário não pedir código, não responda com código por padrão; ofereça no máximo um pseudo-exemplo opcional ao final.

Regra de confiabilidade:
- Nunca presuma contexto operacional, métricas, volumes ou resultados.
- Se algo não estiver explicitamente descrito nos documentos ou na pergunta,
  trate como desconhecido.
- Prefira assumir incerteza a fornecer uma resposta imprecisa.

Encerramento:
- Sempre conclua com uma recomendação orientada à decisão de negócio.
- Evite perguntas genéricas.
- Só faça perguntas ao usuário se isso destravar uma escolha prática
  (ex: orçamento, volume de clientes, restrição operacional).

Histórico da conversa:
{chat_history}

Contexto (documentos relevantes):
{context}

Pergunta:
{question}

Resposta:
""",
            )

            # Cadeia (SEM fontes)
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=False,
                output_key="answer",
            )

            # Rodar consulta
            resultado = chain.invoke({"question": nova_pergunta})

            # Guardar resposta (opcional)
            st.session_state["last_answer"] = resultado.get("answer", "")

            # Rerun para renderizar no histórico
            st.rerun()

        except Exception as e:
            st.error(f"Erro ao gerar resposta: {e}")

import streamlit as st
import joblib
import spacy
import pandas as pd

# =========================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================
st.set_page_config(page_title="Triagem de chamados", page_icon="💬")

# =========================================
# CARREGAMENTO DE RECURSOS (COM CACHE)
# =========================================


@st.cache_resource
def carregar_modelo():
    """
    Carrega o modelo de Machine Learning treinado (.pkl)
    O cache evita recarregar a cada interação.
    """
    modelo = joblib.load("modelo_triagem_suporte.pkl")

    if modelo is None:
        raise ValueError("Erro ao carregar o modelo de ML")

    return modelo


@st.cache_resource
def carregar_nlp():
    """
    Carrega o modelo de linguagem do spaCy (português)
    """
    nlp_model = spacy.load("pt_core_news_sm")

    if nlp_model is None:
        raise ValueError("Erro ao carregar o modelo NLP")

    return nlp_model


# =========================================
# INICIALIZAÇÃO DOS MODELOS
# =========================================
try:
    modelo = carregar_modelo()
    nlp = carregar_nlp()  # <-- CORRETO (com parênteses)

except Exception as e:
    st.error("Erro ao carregar recursos. Execute o script de treinamento.")
    st.error(str(e))
    st.stop()


# =========================================
# FUNÇÃO PRINCIPAL DE PROCESSAMENTO
# =========================================
def analisar_chamado(texto_usuario):
    """
    Processa o texto do usuário e retorna:
    - Categoria prevista
    - Confiança
    - Entidades reconhecidas
    """

    # ========================
    # Etapa 1: NLP (spaCy)
    # ========================
    doc = nlp(texto_usuario)

    # Extração de entidades nomeadas
    entidades = [(ent.text, ent.label_) for ent in doc.ents]

    # ========================
    # Etapa 2: Limpeza do texto
    # ========================
    texto_limpo = " ".join(token.lemma_.lower() for token in doc if not token.is_punct)

    # ========================
    # Etapa 3: Predição ML
    # ========================
    categoria_predita = modelo.predict([texto_limpo])[0]

    probs = modelo.predict_proba([texto_limpo])[0]
    confianca = max(probs) * 100

    return categoria_predita, confianca, entidades


# =========================================
# INTERFACE (STREAMLIT)
# =========================================

st.title("Triagem de suporte")
st.markdown("Descreva o problema em poucas palavras.")

# Histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir mensagens antigas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada do usuário
if prompt := st.chat_input("Ex: O servidor AWS parou de responder..."):

    # Exibir mensagem do usuário
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Processar texto
    categoria, confianca, ents = analisar_chamado(prompt)

    # Montar resposta
    resposta_md = f"""
**Análise do chamado:**
**Categoria:** `{categoria}`  
**Confiança:** `{confianca:.2f}%`
"""

    # Adicionar entidades (se existirem)
    if ents:
        resposta_md += "\n\n**Entidades detectadas:**"
        for ent in ents:
            resposta_md += f"\n- *{ent[0]}* ({ent[1]})"

    # Exibir resposta
    #with st.chat_message("assistant"):
        #st.markdown(resposta_md)

    st.session_state.messages.append({"role": "assistant", "content": resposta_md})

    #ações automáticas por categoria
    acoes = {
        "Infraestrutura": " Encaminhando para equipe N2",
        "Acesso": "Verificando logd de autenticação",
        "Hardware": " Abrindo ordem de serviço.",
            "Software": " Verificando disponibilidade de licenças."
    }
        
    #adicionar as ações sugestidas com base na categoria
    resposta_md += f"\n\n **Ação:**{acoes.get(categoria,'Triagem manual necessária.')}"

    #3. exibir a resposta do assitente

    with st.chat_message('assistant'):
        st.markdown(resposta_md)


    st.session_state.messages.append({
        "role": "assistant",
        "content":resposta_md
    })
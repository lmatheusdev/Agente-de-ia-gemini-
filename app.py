import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, TypedDict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import StateGraph, START, END




# Carrega as variáveis do arquivo .env.
load_dotenv()

# Acessa a chave da API e a guarda.
apiKey = os.getenv("GOOGLE_API_KEY")

TRIAGEM_PROMPT = ( # define regras de triagem (como a ia deve responder)
    "Você é um triador de Service Desk para políticas internas da empresa RDF telecom. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").'
    "Analise a mensagem e decida a ação mais apropriada."
)

class TriagemOut(BaseModel): # define o formato do JSON qeu a ia deve retornar
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"] # define os valores aceitos
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"] #
    campos_faltantes: List[str] = Field(default_factory=list) # cria uma lista

llm_triagem = ChatGoogleGenerativeAI( # cria a ia
    model="gemini-2.5-flash", # modelo da ia
    temperature=0.0, # temperatura da ia
    api_key= apiKey # chave da api
)

triagem_chain = llm_triagem.with_structured_output(TriagemOut) # cria o agente de triagem e atribui configuracoes de saida

def triagem(mensagem: str) -> Dict: # cria a funcao de triagem reutilizavel
    saida: TriagemOut = triagem_chain.invoke([ # TriagemOut define o tipo de sáida / triagem_chain invoca a ia (valor da variavel)
        SystemMessage(content=TRIAGEM_PROMPT), # atribui o prompt de triagem a mensagem de resposta da ia (configura a ia)
        HumanMessage(content=mensagem) 
    ])

    return saida.model_dump() # garante que a saida seja um dicionario

testes1 = ["Posso reembolsar a internet?",
          "Quero mais 5 dias de trabalho remoto. Como faço?"]
"""
for msg_teste in testes:
    print(f"Pergunta: {msg_teste}\n -> Resposta: {triagem(msg_teste)}\n") # passa as perguntas uma a uma para a funçao
"""
docs = [] # Cria uma lista vazia chamada docs que será usada para armazenar os documentos carregados.

for n in Path("./docs/").glob("*.pdf"): # Percorre todos os arquivos PDF na pasta "docs"
    try:
        loader = PyMuPDFLoader(str(n)) # Converte o objeto n(documento) para uma string e armezana no loader
        docs.extend(loader.load()) # Carrega o conteúdo do arquivo PDF usando loader.load() e adiciona os documentos 
        # carregados à lista docs.
        # .extend() = método que adiciona elementos ao final de uma lista existente (semelhante ao .push() em JS)
    except Exception as e:
        print(f"Erro ao carregar arquivo {n.name}: {e}")

print(f"Total de documentos carregados: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30) # define o tamanho dos chunks
# chunk_size = tamanho dos chunks
# chunk_overlap = tamanho do overlap entre os chunks

chunks = splitter.split_documents(docs) # divide os documentos em chunks

embeddings = GoogleGenerativeAIEmbeddings( # chama o modelo de ia responsavel por gerar os embeddings
    model="models/gemini-embedding-001",
    google_api_key= apiKey
)

vectorstore = FAISS.from_documents(chunks, embeddings) 
# FAISS = faz a relação de similaridade entre os embeddings.
# embeddings = chama o modelo de ia responsavel por gerar os embeddings

# retriever = configura como sera feita a busca de similaridade, e as armazena
retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", # define o tipo de busca
                                     search_kwargs={"score_threshold":0.3, "k": 4}) 
                                    # define o limite de busca (intervalo de similaridade)

prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Assistente de Políticas Internas (RH/IT) da empresa RDF telecom. "
     "Responda SOMENTE com base no contexto fornecido. "
     "Se não houver base suficiente, responda apenas 'Não sei'."),

    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag) # conecta o llm com o rag()

# Formatadores
import re, pathlib

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1: break
    if pos == -1: pos = 0
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source","")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]

def perguntar_politica_RAG(pergunta: str) -> Dict:
    docs_relacionados = retriever.invoke(pergunta) # busca os documentos relacionados

    if not docs_relacionados: # se nao houver documentos relacionados
        return {"answer": "Não sei.",
                "citacoes": [],
                "contexto_encontrado": False}

    answer = document_chain.invoke({"input": pergunta,
                                    "context": docs_relacionados})

    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei.",
                "citacoes": [],
                "contexto_encontrado": False}

    return {"answer": txt, # se houver documentos relacionados, retorna a resposta
            "citacoes": formatar_citacoes(docs_relacionados, pergunta),
            "contexto_encontrado": True}

for msg_teste in testes1:
    resposta = perguntar_politica_RAG(msg_teste)
    print(f"PERGUNTA: {msg_teste}")
    print(f"RESPOSTA: {resposta['answer']}")
    if resposta['contexto_encontrado']:
        print("CITAÇÕES:")
        for c in resposta['citacoes']:
            print(f" - Documento: {c['documento']}, Página: {c['pagina']}")
            print(f"   Trecho: {c['trecho']}")
        print("------------------------------------")

# O AgentState funciona como cérebro temporário do agente. Ele armazena as informações que o agente precisa para executar as tarefas.
class AgentState(TypedDict, total = False): # usa TypedDict para criar um tipo de dados personalizado
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str

def node_triagem(state: AgentState) -> AgentState: 
    print("Executando nó de triagem...")
    return {"triagem": triagem(state["pergunta"])}

def node_auto_resolver(state: AgentState) -> AgentState: 
    print("Executando nó de auto_resolver...")
    resposta_rag = perguntar_politica_RAG(state["pergunta"])

    update: AgentState = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes", []),
        "rag_sucesso": resposta_rag["contexto_encontrado"],
    }

    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"

    return update

def node_pedir_info(state: AgentState) -> AgentState:
    print("Executando nó de pedir_info...")
    faltantes = state["triagem"].get("campos_faltantes", [])
    if faltantes:
        detalhe = ",".join(faltantes)
    else:
        detalhe = "Tema e contexto específico"

    return {
        "resposta": f"Para avançar, preciso que detalhe: {detalhe}",
        "citacoes": [],
        "acao_final": "PEDIR_INFO"
    }

def node_abrir_chamado(state: AgentState) -> AgentState:
    print("Executando nó de abrir_chamado...")
    triagem = state["triagem"]

    return {
        "resposta": f"Abrindo chamado com urgência {triagem['urgencia']}. Descrição: {state['pergunta'][:140]}",
        "citacoes": [],
        "acao_final": "ABRIR_CHAMADO"
    }

KEYWORDS_ABRIR_TICKET = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"] 

def decidir_pos_triagem(state: AgentState) -> str:
    print("Decidindo após a triagem...")
    decisao = state["triagem"]["decisao"]

    if decisao == "AUTO_RESOLVER": return "auto"
    if decisao == "PEDIR_INFO": return "info"
    if decisao == "ABRIR_CHAMADO": return "chamado"

def decidir_pos_auto_resolver(state: AgentState) -> str:
    print("Decidindo após o auto_resolver...")

    if state.get("rag_sucesso"):
        print("Rag com sucesso, finalizando o fluxo.")
        return "ok"

    state_da_pergunta = (state["pergunta"] or "").lower()

    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
        print("Rag falhou, mas foram encontradas keywords de abertura de ticket. Abrindo...")
        return "chamado"

    print("Rag falhou, sem keywords, vou pedir mais informações...")
    return "info"

workflow = StateGraph(AgentState) # define a estrutura do fluxo (gráfico de execução)

workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("abrir_chamado", node_abrir_chamado)

workflow.add_edge(START, "triagem")
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info",
    "chamado": "abrir_chamado"
})

workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "info": "pedir_info",
    "chamado": "abrir_chamado",
    "ok": END
})

workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_chamado", END)

grafo = workflow.compile()

testes = ["Posso reembolsar a internet?",
          "Quero mais 5 dias de trabalho remoto. Como faço?",
          "Posso reembolsar cursos ou treinamentos da Alura?",
          "É possível reembolsar certificações do Google Cloud?",
          "Posso obter o Google Gemini de graça?",
          "Qual é a palavra-chave da aula de hoje?",
          "Quantas capivaras tem no Rio Pinheiros?"]

for msg_test in testes:
    resposta_final = grafo.invoke({"pergunta": msg_test})

    triag = resposta_final.get("triagem", {})
    print(f"PERGUNTA: {msg_test}")
    print(f"DECISÃO: {triag.get('decisao')} | URGÊNCIA: {triag.get('urgencia')} | AÇÃO FINAL: {resposta_final.get('acao_final')}")
    print(f"RESPOSTA: {resposta_final.get('resposta')}")
    if resposta_final.get("citacoes"):
        print("CITAÇÕES:")
        for citacao in resposta_final.get("citacoes"):
            print(f" - Documento: {citacao['documento']}, Página: {citacao['pagina']}")
            print(f"   Trecho: {citacao['trecho']}")

    print("------------------------------------")
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


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

testes = ["Posso reembolsar a internet?",
          "Quero mais 5 dias de trabalho remoto. Como faço?"]

for msg_teste in testes:
    print(f"Pergunta: {msg_teste}\n -> Resposta: {triagem(msg_teste)}\n") # passa as perguntas uma a uma para a funçao

docs = [] # Cria uma lista vazia chamada docs que será usada para armazenar os documentos carregados.

for n in Path("./docs/").glob("*.pdf"): # Percorre todos os arquivos PDF na pasta "docs"
    try:
        loader = PyMuPDFLoader(str(n)) # Converte o objeto n(documento) para uma string e armezana no loader
        docs.extend(loader.load()) # Carrega o conteúdo do arquivo PDF usando loader.load() e adiciona os documentos 
        # carregados à lista docs.
        # .extend() = método que adiciona elementos ao final de uma lista existente (semelhante ao .push() em JS)
        print(f"Carregado com sucesso arquivo {n.name}")
    except Exception as e:
        print(f"Erro ao carregar arquivo {n.name}: {e}")

print(f"Total de documentos carregados: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30) # define o tamanho dos chunks
# chunk_size = tamanho dos chunks
# chunk_overlap = tamanho do overlap entre os chunks

chunks = splitter.split_documents(docs) # divide os documentos em chunks

for chunk in chunks: # imprime os chunks
    print(chunk)
    print("------------------------------------")

embeddings = GoogleGenerativeAIEmbeddings( # chama o modelo de ia responsavel por gerar os embeddings
    model="models/gemini-embedding-001",
    google_api_key= apiKey
)

vectorstore = FAISS.from_documents(chunks, embeddings) 
# FAISS = faz a relação de similaridade entre os embeddings.
# embeddings = chama o modelo de ia responsavel por gerar os embeddings

# retriever = configura como sera feita a busca de similaridade
retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", # define o tipo de busca
                                     search_kwargs={"score_threshold":0.3, "k": 4}) 
                                    # define o limite de busca (intervalo de similaridade)



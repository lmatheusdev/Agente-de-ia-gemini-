import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from langchain_core.messages import SystemMessage, HumanMessage

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

    return saida.model_dump() # fgarante que a saida seja um dicionario

testes = ["Posso reembolsar a internet?",
          "Quero mais 5 dias de trabalho remoto. Como faço?",
          "Quantas uniformes eu tenho?"]

for msg_teste in testes:
    print(f"Pergunta: {msg_teste}\n -> Resposta: {triagem(msg_teste)}\n") # passa as perguntas uma a uma para a funçao
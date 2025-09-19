# Aula 1

## Agenstes de Ia

  Um agente de IA (Inteligência Artificial) é um sistema de software autônomo que usa a IA para interagir
  com o ambiente, tomar decisões, aprender e realizar tarefas para atingir objetivos predefinidos em nome
  dos utilizadores ou de um sistema maior.
  
## langchain

LangChain é um framework de código aberto que facilita a criação de aplicações que utilizam Grandes Modelos de Linguagem (LLMs),
oferecendo um conjunto de ferramentas modulares e integrações para conectar LLMs a diversas fontes de dados, como documentos e APIs.
Ele permite construir fluxos de trabalho complexos para aplicações como chatbots, agentes de IA autônomos e sistemas de Geração
Aumentada de Recuperação (RAG).

## Prompts

Prompt é uma sequência de instruções para um modelo de linguagem de IA (LLM) ou um agente de IA.

### Boas práticas

- O prompt deve ser claro e conciso, evitando ambiguidades.
- Deve estabelecer uma personalidade clara e um objetivo para o agente de IA.
- Deve estabelecer o formato e a estrutura de resposta de maneira clara e objetiva.
- Defina um contexto claro para o agente de IA, com exemplos de situações e respostas esperadas.
- Use exemplos concretos e relevantes para ilustrar o contexto e a resposta esperada. Quanto mais exemplos melhor!

# Aula 2

## Splitter / Chunks

O splitter divide um texto em partes menores para facilitar a compreensão e a manipulação dos dados.

## Embeddings

são representações numéricas (vetores) de dados como texto, imagens, áudio ou outros objetos, criadas por modelos de aprendizado de
máquina para capturar o seu significado e as relações entre eles. Essa forma matemática permite que sistemas de IA entendam e processem
dados complexos, permitindo, por exemplo, a busca por similaridade (encontrar conteúdos parecidos), sistemas de recomendação e tradução
de idiomas. Objetos com significados semelhantes são representados por vetores próximos uns dos outros num espaço multidimensional.

Resumidamente: a ia cria uma relação de semelhança semântica entre dados, definindo valores númericos para cada um dos dados. Podendo
assim fazer relações de similaridade entre os dados, sejam eles palavras, textos, imagens, etc.

No contexto do agente que estamos criando: a ia cria embeddings para cada chunk do documento, permitindo que ela possa entender
relações semânticas entre eles.

## FAISS - Facebook AI Similarity Search

O Faiss é uma biblioteca de código aberto projetada para pesquisa de similaridade eficiente e agrupamento de vetores densos, permitindo
aplicativos como sistemas de recomendação e pesquisa de imagens.

Ela faz a relação de similaridade entre os embeddings.

## RAG - Retriever Augmented Generation

RAG (Retrieval Augmented Generation) é uma técnica utilizada para ampliar a capacidade de resposta de LLMs, combinando o conhecimento
interno do modelo de linguagem com sistemas de recuperação de informações.

Ou seja, o modelo busca informações relevantes em bases de dados externas como bancos de dados ou documentos organizacionais antes de
gerar uma resposta, permitindo acesso a dados atualizados, especializados ou muito específicos sem a necessidade de re-treinar o modelo.

# Aula 3

## Langgraph

LangGraph é uma estrutura (framework) em Python, parte do projeto LangChain, que permite criar sistemas de inteligência artificial (IA)
complexos e com estado, modelando fluxos de trabalho como gráficos interconectados. Ele permite a construção de agentes que raciocinam,
planejam e agem através de múltiplas etapas e interações, sendo ideal para aplicações que exigem memória, tomada de decisão e a
colaboração entre vários agentes.

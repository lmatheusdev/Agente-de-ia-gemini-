# Notas

## Criar o ambiente virtual

  python -m venv .venv

## Ativar o ambiente virtual

  No PowerShell/terminal do VS Code: .\.venv\Scripts\activate

## instalar o pacote dotenv

  python -m pip install python-dotenv

## Verificar instalação

  python -m pip install python-dotenv

## Sobre a pasta .venv (ambiente virtual)

  A .venv é para arquivos do ambiente (executáveis e pacotes).
  Faz com que o projeto rode em um ambiente separado (virtual), onde
  as bibliotecas intaladas são acessiveis somente no projeot corrente,
  não afetando outros projetos.

## Como o load_dotenv() procura o .env

  load_dotenv() sem argumentos tenta carregar .env do diretório onde o processo está rodando

## Check rápido / comandos de depuração úteis

Execute (no terminal dentro da pasta do projeto, com a venv ativada):

```py
  # confirmar qual python está rodando
where python        # Windows
# ou (Linux / macOS)
which python

# confirmar pip/instalação no mesmo python
python -m pip show python-dotenv

# testar import direto
python -c "import dotenv; print(dotenv.__file__)"
```

## O que é o .gitignore

  O .gitignore é um arquivo de configuração usado pelo Git.
  Ele lista arquivos/pastas que não devem ser versionados (ou seja,
  não vão para o repositório remoto no GitHub/GitLab).

ex:.

(digitar dentro do arquivo .gitignore)
Ignorar venv
.venv/

Ignorar arquivos de configuração local
.vscode/

Ignorar chaves e senhas
.env

## Subir o servidor FastAPI

  uvicorn main:app --reload

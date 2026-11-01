# GenAI-LangChain-AgenteAvaliador
Projeto Agente Avaliador - Especialização em IA Generativa - UFPR

`agente_avaliacoes.py` consulta a API de avaliações da Codante e classifica cada
descrição recebida como **positiva** ou **negativa**, exibindo também a confiança
estimada pelo modelo.

Componentes aplicados:
*    PromptTemplate: Padroniza o texto da avaliação enviado ao modelo.
*    ChatPromptTemplate: Organiza regras de sistema, histórico e mensagem atual.
*    ChatGroq: Usa LLM em provedor remoto.
*    StrOutputParser: Extrai a resposta textual da mensagem da LLM.
*    BaseModel e Field (Pydantic): Validam JSON, classe binária e confiança entre 0 e 1.
*    Histórico em CSV e janela deslizante: Persistem as quatro últimas interações válidas.

Preparação do ambiente (Python 3.10 ou superior):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Forneça a API KEY do GROQ no  `.env`:

```dotenv
GROQ_API_KEY=sua_chave_aqui
GROQ_MODEL=qwen/qwen3.8-27b
```

Execução:

```powershell
.\.venv\Scripts\python.exe agente_avaliacoes.py
```
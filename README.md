# LangChain + Groq Practice Project

This project is a hands-on learning path for LangChain using Groq-supported chat models.
It includes focused examples for core topics and one complete proof-of-concept project.

## What You Will Practice

- Calling Groq chat models through LangChain
- Prompt templates
- Output parsers
- LCEL chains
- Conversation memory patterns
- Document loading and chunking
- Embeddings and vector search
- Retrieval-augmented generation
- Tools
- Groq-backed agents
- Streaming
- Structured outputs
- Error handling and retries
- A complete POC research assistant

## Project Layout

```text
langchain-groq-practice/
  examples/              # Small focused examples
  poc_research_agent/    # Complete proof-of-concept project
  data/                  # Sample documents for retrieval examples
  .env.example           # Environment variable template
  requirements.txt       # Python dependencies
```

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file from `.env.example`.

```powershell
Copy-Item .env.example .env
```

4. Add your Groq API key to `.env`.

```text
GROQ_API_KEY=your_groq_api_key_here
```

## Recommended Learning Order

Run the examples in this order:

```powershell
python examples/01_chat_model.py
python examples/02_prompt_templates.py
python examples/03_output_parsers.py
python examples/04_lcel_chains.py
python examples/05_conversation_memory.py
python examples/06_document_loading_chunking.py
python examples/07_embeddings_vector_search.py
python examples/08_rag.py
python examples/09_tools.py
python examples/10_groq_agent.py
python examples/11_streaming.py
python examples/12_structured_output.py
python examples/13_error_handling.py
```

Then run the complete POC:

```powershell
python poc_research_agent/app.py
```

## Groq Models

The examples default to:

```text
llama-3.3-70b-versatile
```

You can change the model with:

```text
GROQ_MODEL=llama-3.1-8b-instant
```

in your `.env` file.

## Notes

The examples are deliberately commented. Read the comments first, then run the files, then modify one thing at a time.

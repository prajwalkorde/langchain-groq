# POC: Internal Research Assistant

This proof-of-concept turns local text files into a simple research assistant.

It demonstrates:

- Loading documents
- Splitting documents
- Creating a vector index
- Retrieval-augmented answering
- Tool use
- A Groq-backed tool-calling agent

## Run

From the project root:

```powershell
python poc_research_agent/app.py
```

Try questions like:

```text
What does InsightPilot do?
What are common project risks for Acme Analytics?
Create a brief implementation plan for InsightPilot.
Calculate the margin for revenue 25000 and cost 17500.
```

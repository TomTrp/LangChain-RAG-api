from fastapi import FastAPI

app = FastAPI(title="LLM/RAG API")

@app.get("/health")
def health():
    return {"status": "ok"}
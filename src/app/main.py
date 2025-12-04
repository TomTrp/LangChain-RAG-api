from fastapi import FastAPI
from app.documents import load_documents
from app.chroma import get_chroma_setting
from app.rag_pipeline import get_retrieval, get_chat_model, ask_with_context

app = FastAPI(title="LLM/RAG API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/documents")
def documents():
    return {"response": load_documents()}

@app.get("/chroma")
def chroma_settings():
    return {"response": get_chroma_setting()}

@app.post("/ask")
def ask(question: str):
    llm = get_chat_model()
    retrieval = get_retrieval(question)
    result = ask_with_context(
        llm=llm,
        question=question,
        retrieved_docs=retrieval
    )

    return {"response": result}
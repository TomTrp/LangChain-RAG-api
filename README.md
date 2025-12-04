# LLM / RAG API with LangChain, HuggingFace & Chroma

This project demonstrates how to build a **Retrieval-Augmented Generation (RAG)** pipeline with **FastAPI, LangChain, HuggingFace**, and **Chroma Vector Database.**
It allows you to upload or store text documents, embed them into a vector store, and query a Large Language Model (LLM) that answers questions **strictly based on your documents**.

## 🔧 Features
- ✅ FastAPI endpoint for querying LLM responses
- 🧩 LangChain integration for document loading, embedding, and retrieval
- 🧠 HuggingFaceEndpoint for both embedding and chat model (no local model required)
- 📂 Chroma Vector Store for document similarity search
- 🚀 Lightweight & Docker-ready (no PyTorch installation needed)

## 🚀 How to Run

### 1. Install Required Tools
- Python 3.10 or higher
- Required Python packages:
  `pip install fastapi uvicorn langchain langchain-community langchain-huggingface langchain-text-splitters pydantic chromadb`
**Note**: This version uses HuggingFace Inference API, so **no local PyTorch or sentence-transformers installation** is required.

### 2. Setup Environment Variables
- Create a .env file in the project root:
```
  HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
  DOCS_PATH=./docs
  CHROMA_HOST=chroma
  CHROMA_PORT=8000
```
  You can get your Hugging Face token here → [HuggingFace](https://huggingface.co/settings/tokens)
  
### 3. Add Your Knowledge Base
- Place your text documents inside the /docs directory.
Example:
```
    docs/
     ├── cake-recipe.txt
     └── smoothie-recipe.txt
```
Each .txt file will be automatically loaded and embedded into Chroma.

### 4. Run FastAPI Server
- Start the FastAPI service:

    `uvicorn app.main:app --reload`
- Open your browser at: 
    [Swagger UI](http://localhost:8000/docs)
- Available endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/documents` | Load and return all documents (raw text) |
| `GET` | `/ask?question=...` | Ask a question — LLM answers based only on the documents |

- Example JSON Request:
  /ask?question=How to make smoothies
```

- Example Response:
```json
{
  "response": "Ingredients
  1 quart strawberries, hulled
  2 fresh peaches - peeled, pitted, and sliced
  1 banana, broken into chunks
  2 cups ice
  1 cup orange-peach-mango juice

Directions
  step 1
   Gather all ingredients.
  step 2
   Combine strawberries, peaches, and banana in a blender; blend until smooth.
  step 3
   Add ice and pour in juice; blend again to desired consistency."
}
```

### 5. (Optional) Run with Docker
- Build and run via Docker Compose:
    `docker compose up --build`
- app → FastAPI + LangChain API (port 8000)
- chroma → Chroma Vector DB (port 8001)
- Visit: `http://localhost:8000/docs`

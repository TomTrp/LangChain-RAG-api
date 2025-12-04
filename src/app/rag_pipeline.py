from langchain.messages import  HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings, ChatHuggingFace
from config import HUGGINGFACEHUB_API_TOKEN
from app.documents import load_documents
from app.chroma import get_chroma_setting, create_vectorstore

def split_docs_chunk(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs_split = splitter.split_documents(docs)
    return docs_split

def get_embedding(query: str):
    hf = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        task="feature-extraction",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    )
    return hf
    # return hf.embed_documents(query)

def get_chat_model():
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        repo_id="deepseek-ai/DeepSeek-R1-0528",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        provider="auto",  # let Hugging Face choose the best provider for you
    )
    chat_model = ChatHuggingFace(llm=llm)
    return chat_model

def get_retrieval(query: str):
    docs = load_documents()
    embedding = get_embedding(query=query)

    client = get_chroma_setting()
    vectorstore = create_vectorstore(client, docs, embedding)
    results = vectorstore.similarity_search(query, k=2)
    return results

def ask_with_context(llm, question, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
    You are an assistant that answers questions strictly and concisely based ONLY on the given context below.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    - Use ONLY the facts from the context.
    - If the context does not contain enough information to answer, reply with:
    "The provided context does not contain enough information to answer this question."
    - Do NOT add your own assumptions or outside knowledge.
    - Keep the answer short and factual.
    """

    messages = [
        SystemMessage(content="You are a retrieval-augmented assistant that answers only from provided context."),
        HumanMessage(content=prompt.strip()),
    ]
    response = llm.invoke(messages)
    return response.content


from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from chromadb.config import Settings
import os

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    huggingfacehub_api_token=api_key,
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

chat_model = ChatHuggingFace(llm=llm)


docs_path = "../docs"
docs = []
for filename in os.listdir(docs_path):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(docs_path, filename), encoding="utf-8")
        docs.extend(loader.load())

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs_split = splitter.split_documents(docs)

# Create embeddings + Chroma vectorstore
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

client_settings = Settings(
    chroma_api_impl="rest",
    chroma_server_host="localhost",
    chroma_server_http_port=8000
)

vectorstore = Chroma.from_documents(
    documents=docs_split,
    embedding=hf,
    client_settings=client_settings  # connect to docker server
)

# Ask a question
query = "Summary test.txt"

# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
# docs_topk = retriever.get(query)
# docs_text = "\n".join([d.page_content for d in docs_topk])

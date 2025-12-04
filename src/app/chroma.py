from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from config import CHROMA_HOST, CHROMA_PORT

def get_chroma_setting():
   client_settings = Settings(
        chroma_api_impl="rest",
        chroma_server_host=CHROMA_HOST,
        chroma_server_http_port=CHROMA_PORT
    )
   return client_settings

def create_vectorstore(client: Settings, documents, embedding):
   vectorstore = Chroma.from_documents(
        # client_settings=client,  # connect to docker server
        documents=documents,
        embedding=embedding,
        collection_name="my_langchain_collection",
    )
   
#    vector_store = Chroma.from_texts(
#         texts=documents,
#         embedding=embedding,
#         collection_name="my_langchain_collection",
#     )
   return vectorstore

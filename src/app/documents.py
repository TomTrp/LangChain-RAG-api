from langchain_community.document_loaders import TextLoader
from config import DOCS_PATH
import os

def load_documents():
    docs = []
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(DOCS_PATH, filename), encoding="utf-8")
            docs.extend(loader.load())
    return docs
from langchain_community.document_loaders import TextLoader
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os
import shutil
from tqdm import tqdm  # add to the top if not already


CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    paths = glob.glob("data/books/*.md")  # or *.txt
    documents = []
    for path in paths:
        loader = TextLoader(path, encoding='utf-8')
        documents.extend(loader.load())
    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 600,
        chunk_overlap = 150,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


    print("⏳ Generating embeddings with Ollama...")

    embedding_function = OllamaEmbeddings(model="mistral")

    # Add progress bar manually (chunk by chunk)
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    embeddings = [embedding_function.embed_query(text) for text in tqdm(texts)]

    # Manually build vectorstore
    from langchain.schema import Document
    new_docs = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas)]

    db = Chroma.from_documents(
        new_docs,
        embedding_function,
        persist_directory=CHROMA_PATH,
    )
    db.persist()

    print(f"✅ Saved {len(new_docs)} chunks to ChromaDB.")

    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()

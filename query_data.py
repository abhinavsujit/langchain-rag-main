import argparse
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = OllamaEmbeddings(model="mistral")  # Change model if needed
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search(query_text, k=3)
    if len(results) == 0:
        print("No relevant chunks found.")
        return
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOllama(model="mistral")  # You can switch to llama3, etc.
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", "N/A") for doc in results]
    print(f"\n🔍 Response:\n{response_text}")
    print(f"\n📚 Sources: {sources}")

    # 🗣️ Speak the response aloud
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(response_text)
    engine.runAndWait()

if __name__ == "__main__":
    main()

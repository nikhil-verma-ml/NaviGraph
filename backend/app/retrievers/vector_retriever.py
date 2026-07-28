# retrievers/vector_retriever.py

import os
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import DATA_DIR


class HybridRetriever:
    """Combines semantic (FAISS) and keyword (BM25) retrieval."""

    def __init__(self, documents=None, vector_store_path=None, k=5):
        if vector_store_path is None:
            vector_store_path = str(DATA_DIR / "vector_store")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.vector_store_path = vector_store_path

        if documents:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.bm25_retriever = BM25Retriever.from_documents(documents)
        else:
            # Check if vector store files exist. If not, auto-seed with default documentation
            index_file = os.path.join(vector_store_path, "index.faiss")
            if not os.path.exists(index_file):
                print(f"[VectorRetriever] Vector store index not found at {vector_store_path}. Creating a default seeded index.")
                from langchain_core.documents import Document
                os.makedirs(vector_store_path, exist_ok=True)
                default_docs = [
                    Document(page_content="Our Agentic RAG system is built using LangGraph, FastAPI, and Streamlit. It uses a single agent node with a ReAct loop. Rather than using external CRAG or Self-RAG evaluator nodes, quality control is handled entirely in the agent's system prompt to maintain agentic autonomy.", metadata={"source": "architecture_guide"}),
                    Document(page_content="The system uses a hybrid retriever combining FAISS (semantic search) and BM25 (keyword search) using an EnsembleRetriever, with weights 0.6 for FAISS and 0.4 for BM25.", metadata={"source": "retriever_guide"}),
                    Document(page_content="Conversation memory is persisted across turns using a SQLite checkpointer (SqliteSaver). Long-term session tracking (like session titles and lists) is managed in a separate SQLite database called SessionStore.", metadata={"source": "memory_guide"}),
                    Document(page_content="The primary model is Google Gemini (gemini-2.0-flash). If Gemini fails or hits rate limits, the system automatically falls back to Groq running llama-3.3-70b-versatile, ensuring high availability.", metadata={"source": "llm_guide"})
                ]
                self.vector_store = FAISS.from_documents(default_docs, self.embeddings)
                self.vector_store.save_local(vector_store_path)
                self.bm25_retriever = BM25Retriever.from_documents(default_docs)
            else:
                self.vector_store = FAISS.load_local(
                    vector_store_path, self.embeddings, allow_dangerous_deserialization=True
                )
                all_docs = list(self.vector_store.docstore._dict.values())
                self.bm25_retriever = BM25Retriever.from_documents(all_docs)

        self.bm25_retriever.k = k
        semantic_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25_retriever, semantic_retriever],
            weights=[0.4, 0.6]
        )

    def retrieve(self, query: str) -> list:
        try:
            return self.ensemble.invoke(query)
        except Exception as e:
            print(f"[VectorRetriever] Retrieval failed: {e}")
            return []

    def save_index(self):
        self.vector_store.save_local(self.vector_store_path)


_retriever_instance = None

def get_vector_retriever() -> HybridRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance
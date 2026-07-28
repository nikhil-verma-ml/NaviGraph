# retrievers/web_retriever.py

import os
from langchain_tavily import TavilySearch
from langchain_core.documents import Document


class WebRetriever:
    """Fetches real-time information from the web using Tavily."""

    def __init__(self, max_results: int = 5):
        self.search_tool = TavilySearch(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=max_results
        )

    def retrieve(self, query: str) -> list[Document]:
        try:
            raw_results = self.search_tool.invoke({"query": query})
            return self._to_documents(raw_results.get("results", []))
        except Exception as e:
            print(f"[WebRetriever] Search failed: {e}")
            return []

    def _to_documents(self, raw_results: list[dict]) -> list[Document]:
        docs = []
        for result in raw_results:
            docs.append(
                Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source": result.get("url", "unknown"),
                        "title": result.get("title", ""),
                    }
                )
            )
        return docs


_web_retriever_instance = None

def get_web_retriever() -> WebRetriever:
    global _web_retriever_instance
    if _web_retriever_instance is None:
        _web_retriever_instance = WebRetriever()
    return _web_retriever_instance
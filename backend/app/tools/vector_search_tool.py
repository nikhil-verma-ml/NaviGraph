from langchain_core.tools import tool
from retrievers.vector_retriever import get_vector_retriever

@tool
def vector_search(query : str) -> str:
    """Search the internal Knowledge base for domain-specific information,
    Use this for questions that are likely covered indexed documents such as reference 
    material , study content, or internal knowledge.

    Args:
        query : The search query to look up in the knowledge base.
    
    Returns :
        Relevant document excerpts as fromatted text or a message if nothing found.
    """

    retriever = get_vector_retriever()
    docs = retriever.retrieve(query)

    if not docs:
        return "No relevant information found in the knowledge base."
    
    formatted = []

    for i ,doc in enumerate(docs,1):
        source = doc.metadata.get("source","unknown")
        formatted.append(f"[Result {i} (source : {source})\n {doc.page_content}]")
    
    return "\n\n".join(formatted)
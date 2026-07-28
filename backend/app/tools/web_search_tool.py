from langchain_core.tools import tool
from retrievers.web_retriever import get_web_retriever

@tool
def web_search(query : str) -> str:
    """Search the internet for real-time or current information.,
    Use this for questions about recent events, current data or anything unlikely to be in the internal knowledge.

    Args:
        query : The search query to look up on the web.
    
    Returns :
        Relevant web search as formatted text or a message if nothing found.
    """

    retriever = get_web_retriever()
    docs = retriever.retrieve(query)

    if not docs:
        return "No relevant information found on the web."
    
    formatted = []

    for i ,doc in enumerate(docs,1):
        source = doc.metadata.get("source","unknown")
        title = doc.metadata.get("title","")
        formatted.append(f"[Result {i} {title} (source : {source})\n {doc.page_content}]")
    
    return "\n\n".join(formatted)
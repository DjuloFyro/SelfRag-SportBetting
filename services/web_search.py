from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults


def web_search(state: dict) -> dict:
    """
    Perform web search using the user question.

    Args:
        state: The state of the conversation.

    Returns:
        state(dict): New key added to the state is `documents` which contains the retrieved documents
    """
    web_search_tool = TavilySearchResults()
    question = state["question"]

    # Perform web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([doc["content"] for doc in docs])
    web_results_docs = Document(page_content=web_results)

    return {"documents": web_results_docs, "question": question}

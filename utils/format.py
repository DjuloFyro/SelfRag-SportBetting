def format_docs(docs: list) -> str:
    """
    Format a list of documents into a single string.

    Args:
        docs: The list of documents to format.

    Returns:
        str: The formatted documents.
    """
    return "\n\n".join([doc.page_content for doc in docs])

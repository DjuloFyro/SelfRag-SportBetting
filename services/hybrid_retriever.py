from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class HybridRetriever:
    """
    A retriever that combines the results of multiple retrievers.
    """

    def __init__(self, documents: list[Document]):
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        doc_splits = text_splitter.split_documents(documents)

        # Create embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(doc_splits, embedding_model)

        # TODO: Do with multiple retriever (BM25 retriever)
        retriever = vectorstore.as_retriever()

        self.retriever = retriever
        # self.weights = [0.7, 0.3]

    def retrieve(self, state: dict) -> dict:
        """
        Retrieve documents from the vectorstore based on a user query.

        Args:
            state: The state of the conversation.

        Returns:
            state(dict): New key added to the state is `retrieved_documents` which contains the retrieved documents.
        """
        question = state["question"]
        documents = self.retriever.invoke(question)

        return {"documents": documents, "question": question}

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class LLMDocRelevanceGrader:
    """
    Class to define the llm that grades the relevance of a document to a user question.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id> You are grader assessing the relevance of a retrieved document to a user question.
            If the document contains keywords related to the question, assign it as relevant. It does not need to be an stringent test. The goal is to filter out erroneous retrievals.\n
            Give a binary score `yes` or `no` to indicate if the document is relevant to the question.\n
            Provide the binary score in a JSON format with a single key `score` and no premable or explanation. <|eot_id|><start_header_id>assistant<end_header_id>
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} <|eot_id|><start_header_id>assistant<end_header_id>
            """,
            input_variable=["question", "document"],
        )

        self.retrieval_grader = self.prompt | self.llm | JsonOutputParser()

    def grade_documents(self, state: dict) -> dict:
        """
        Determine if the retrieved documents are relevant to the user question
        and if a web search is needed.

        Args:
            state: The state of the conversation.

        Returns:
            state(dict): Update documents in state with only relevant ones.
        """
        question = state["question"]
        documents = state["documents"]

        # Score each document
        filtered_docs = []

        for doc in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": doc.page_content}
            )
            if score["score"] == "yes":
                filtered_docs.append(doc)
            else:
                continue

        return {"documents": filtered_docs, "question": question}

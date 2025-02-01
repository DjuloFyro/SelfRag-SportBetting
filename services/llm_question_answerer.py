from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from utils.format import format_docs


class LLMQuestionAnswerer:
    """
    Class to define the llm that answer user questions.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id> You are an assistant for questions answering-tasks.
            Use the following pieces of retrieved information to answer the user question. If you don't know the answer, just say so.
            Use three sentences maximum to answer the question and keep the answer concise. <|eot_id|><start_header_id>assistant<end_header_id>
            Question to answer: {question}
            Context: {context}
            Answer: <|eot_id|><start_header_id>assistant<end_header_id>
            """,
            input_variable=["question", "document"],
        )

        self.question_answerer = self.prompt | self.llm | StrOutputParser()

    def answer_question(self, state: dict) -> dict:
        """
        Generate an answer to a user question based on the retrieved documents.

        Args:
            state: The state of the conversation.

        Returns:
            state(dict): New key added to the state is `generation` which contains the generated answer.
        """
        question = state["question"]
        documents = state["documents"]

        if not isinstance(documents, list):
            documents = [documents]

        # Rag generation
        generation = self.question_answerer.invoke(
            {"question": question, "context": format_docs(documents)}
        )

        return {"documents": documents, "question": question, "generation": generation}

    def decide_to_answer(self, state: dict) -> str:
        """
        Decide whether to generate an answer, or re-generate the question.

        Args:
            state: The state of the conversation.

        Returns:
            state(dict): Update state with the decision to generate an answer.
        """
        filtered_documents = state["documents"]

        if not filtered_documents:
            return "websearch"
        else:
            return "generate"

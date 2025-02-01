from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class LLMAnswerGrader:
    """
    Class to define the llm that grades the hallucination of an answer.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Prompt to grade the hallucination of the answer
        self.prompt_halucination_grader = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id> You are a grader assessing whether an answer is
            grounded in / supported by a set of facts. Give a binary score `yes` or `no` to indicate whether the answer is grounded in / supported by facts.
            Provide the binary score in a JSON format with a single key `score` and no premable or explanation. <|eot_id|><start_header_id>assistant<end_header_id>
            Here is the facts:
            \n ------ \n
            {documents}
            \n ------ \n
            Here is the answer: {generation} <|eot_id|><start_header_id>assistant<end_header_id>
            """,
            input_variable=["documents", "generation"],
        )
        self.hallucination_grader = (
            self.prompt_halucination_grader | self.llm | JsonOutputParser()
        )

        # Prompt to grade the answer
        self.prompt_answer_grader = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id> You are a grader assessing whether an answer is
            useful to resolve a question. Give a binary score `yes` or `no` to indicate whether the answer is useful to resolve the question.
            Provide the binary score as a JSON with a single key `score` and no preamble or explanation. <|eot_id|><start_header_id>assistant<end_header_id>
            Here is the answer:
            \n ------ \n
            {generation}
            \n ------ \n
            Here is the question: {question} <|eot_id|><start_header_id>assistant<end_header_id>
            """,
            input_variable=["question", "generation"],
        )
        self.answer_grader = self.prompt_answer_grader | self.llm | JsonOutputParser()

    def grade_the_answer(self, state: dict) -> str:
        """
        Grade the generated answer against the relevant documents and the user question.

        Args:
            state: The state of the conversation.

        Returns:
            srting: The grade of the answer.
        """
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )

        grade = score["score"]

        # TODO: be more strict about odds related questions

        # check hallucination
        if grade == "yes":
            score = self.answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            grade = score["score"]

            if grade == "yes":
                return "useful"
            else:
                return "not useful"

        else:
            return "not supported"

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class LLMQuestionRouter:
    """
    Class to define the llm that route user questions to the appropriate tool.
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing
            a user question to a vectorstore or web search. Use the vectorstore for questions related to sports betting, football (soccer), team odds, match predictions, and bookmaker analysis.
            You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web search. Give a binary choice
            `vectorstore` or `websearch` based on the question. Return a JSON with a single key `datasource` and no premable or explanation.
            Question to route: {question} <|eot_id|><start_header_id>assistant<end_header_id>""",
            input_variable=["question"],
        )

        self.question_router = self.prompt | self.llm | JsonOutputParser()

    def route_question(self, question: str) -> str:
        """
        Route the user question to the appropriate tool. e.g. vectorstore or web search.

        Args:
            question: The user question.

        Returns:
            (str): The appropriate tool to use.
        """
        source = self.question_router.invoke({"question": question})

        if source["datasource"] == "vectorstore":
            return "vectorstore"
        elif source["datasource"] == "websearch":
            return "websearch"
        else:
            raise ValueError("LLMQuestionRouter: Invalid datasource.")

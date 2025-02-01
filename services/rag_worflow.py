from langgraph.graph import END, START, StateGraph

from services.hybrid_retriever import HybridRetriever
from services.llm_answer_grader import LLMAnswerGrader
from services.llm_doc_relevance_grader import LLMDocRelevanceGrader
from services.llm_question_answerer import LLMQuestionAnswerer
from services.llm_question_routing import LLMQuestionRouter
from services.web_search import web_search
from utils.models import GraphState


def rag_workflow(retriever: HybridRetriever) -> StateGraph:
    """
    Define the RAG workflow.

    Args:
        retriever(HybridRetriever): The retriever to use.

    Returns:
        workflow(StateGraph): The RAG workflow.
    """

    # Define the LLM models
    question_router = LLMQuestionRouter()
    retrieval_grader = LLMDocRelevanceGrader()
    answer_grader = LLMAnswerGrader()
    question_answerer = LLMQuestionAnswerer()

    # Define the GraphState class
    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("web_search", web_search)
    workflow.add_node("retrieve", retriever.retrieve)
    workflow.add_node("grade_documents", retrieval_grader.grade_documents)
    workflow.add_node("generate", question_answerer.answer_question)

    # build graph logic
    workflow.add_conditional_edges(
        START,
        question_router.route_question,
        {
            "vectorstore": "retrieve",
            "websearch": "web_search",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        question_answerer.decide_to_answer,
        {
            "generate": "generate",
            "websearch": "web_search",
        },
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_conditional_edges(
        "generate",
        answer_grader.grade_the_answer,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "web_search",
        },
    )

    return workflow

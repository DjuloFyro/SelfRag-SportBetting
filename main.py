import asyncio

from services.crawler import crawl_with_llm
from services.hybrid_retriever import HybridRetriever
from services.rag_worflow import rag_workflow


async def main():
    question = "What are the odds for the match PSG vs marseille?"

    # TODO: Generic code to handle all bookmakers
    # Crawl the web
    scraped_documents = await crawl_with_llm("betclic")

    # TODO: Persistent VectorStore on all books

    # Define the retriever
    retriever = HybridRetriever(scraped_documents)

    # Define the rag workflow
    workflow = rag_workflow(retriever)

    # Compile the workflow
    compiled_workflow = workflow.compile()

    # Run the workflow with an input question
    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
    }

    # Execute the workflow
    final_state = None
    for output in compiled_workflow.stream(initial_state):
        final_state = output  # Store the latest state
        print(final_state)
        print("===" * 20)

    # TODO: replace with streamlit UI
    # Print only the final generated response
    if final_state and final_state["generate"]["generation"]:
        print("Final generated response:")
        print(final_state["generate"]["generation"])
    else:
        print("We cannot find an answer to your question.")


if __name__ == "__main__":
    asyncio.run(main())

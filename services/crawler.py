import json
from datetime import datetime

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from langchain.schema import Document

from config.settings import BOOKMAKER_LINKS, CSS_SELECTORS, OPENAI_API_KEY
from utils.models import Match


async def crawl_with_llm(bookmaker: str) -> list[Document]:
    """
    Crawl a bookmaker website using LLM extraction strategy.

    Args:
        bookmaker: The name of the bookmaker to crawl.

    Returns:
        (list[Document]): A list of documents containing extracted match data.
    """

    # 1. Define the LLM extraction strategy
    llm_strategy = LLMExtractionStrategy(
        provider="openai/gpt-4o-mini",
        api_token=OPENAI_API_KEY,
        schema=Match.model_json_schema(),
        extraction_type="schema",
        instruction=f"""
        Extract all pre-match (non-live) data from the provided HTML. For each match, return the following details in JSON format according to the schema:

        1. `start_date`: Match date and time in the format `YYYY-MM-DD HH:MM:SS` (relative to `{datetime.now()}`).
        2. `home_team`: Home team name in lowercase with spaces replaced by underscores.
        3. `away_team`: Away team name in lowercase with spaces replaced by underscores.
        5. `odd_home`: Odds for the home team winning.
        6. `odd_draw`: Odds for a draw.
        7. `odd_away`: Odds for the away team winning.
        """,
        chunk_token_threshold=1000,
        overlap_rate=0.1,
        apply_chunking=True,
    )

    # 2. Build the crawler config
    crawl_config = CrawlerRunConfig(
        css_selector=CSS_SELECTORS[bookmaker],
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=llm_strategy,
        wait_for=CSS_SELECTORS[bookmaker],
        delay_before_return_html=5,
    )

    # 3. Create a browser config if needed
    browser_cfg = BrowserConfig(headless=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # 4. Crawl the bookmaker website
        result = await crawler.arun(url=BOOKMAKER_LINKS[bookmaker], config=crawl_config)

        match_data = json.loads(result.extracted_content)

        # 5. Convert the data to documents for further processing
        match_texts = [json.dumps(match) for match in match_data]
        documents = [Document(page_content=text) for text in match_texts]

        return documents

    raise ValueError("No data extracted.")

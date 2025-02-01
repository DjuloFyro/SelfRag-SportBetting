import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MODEL_NAME = "gpt-4o-mini"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50

BOOKMAKER_LINKS = {
    "betclic": "https://www.betclic.fr/football-sfootball",
}

CSS_SELECTORS = {
    "betclic": "div.groupEvents:not(.is-live)",
    "unibet": "#cps-eventsdays-list",
    "betsson": ".betEventList.sbPreLoadable.sbFlexStretch",
    "betstars": "._8fb2e8d[data-testid='sports-match-filtering-widget']",
    "bwin": "ms-main-column.column-wrapper",
}

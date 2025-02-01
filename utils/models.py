from pydantic import BaseModel
from typing_extensions import TypedDict


class Match(BaseModel):
    match_start: str
    home_team: str
    away_team: str
    home_odds: float
    draw_odds: float
    away_odds: float


class GraphState(TypedDict):
    question: str
    documents: list
    generation: str

from typing import Dict, Optional
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
    agent_params: Optional[Dict] = None

class QuestionResponse(BaseModel):
    original_question: str
    refined_question: str
    answer: str
    quality_score: float
    status: str 
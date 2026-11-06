from pydantic import BaseModel, Field


class TextAnalysisRequest(BaseModel):
    user_id: str
    text: str = Field(min_length=1, max_length=5000)
    context: str | None = Field(default=None, max_length=10000)


class TextAnalysisResponse(BaseModel):
    status: str
    risk_level: str
    risk_score: float
    threat_type: str
    recommended_action: str
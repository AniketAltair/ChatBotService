from pydantic import BaseModel
from datetime import datetime

class ChatbotResponse(BaseModel):
    userid: int
    type: str
    answer: dict | str | None
    timestamp: datetime

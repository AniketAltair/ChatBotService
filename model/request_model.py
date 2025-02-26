from pydantic import BaseModel
from datetime import datetime

class ChatbotRequest(BaseModel):
    userid: int
    type: str  # "chatbot" or "info"
    question: str
    timestamp: datetime

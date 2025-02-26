from pydantic import BaseModel
from datetime import datetime

class NewGeneralItemRequest(BaseModel):
    Description : str
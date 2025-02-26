from pydantic import BaseModel
from datetime import datetime

class NewFoodItemRequest(BaseModel):
    Name: str
    Protein: float
    Carbohydrates: float
    Fats: float
    Calories: float
    QuantityType: str
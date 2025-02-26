from fastapi import APIRouter, HTTPException
from model.request_model import ChatbotRequest
from model.response_model import ChatbotResponse
from model.newFoodItem_model import NewFoodItemRequest
from model.newGeneralInfo_model import NewGeneralItemRequest
from service.addNewFood import (add_new_food_item_to_excel,add_new_general_item_to_excel)
from service.answerChatbot import (get_best_match,calculateFinalMacros,calculateFinalMacrosInfo)
from datetime import datetime
import psutil

chatbot_router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

@chatbot_router.post("/ask", response_model=ChatbotResponse)
def ask_chatbot(request: ChatbotRequest):
    if not request.userid or not request.question or not request.type or not request.timestamp:
        raise HTTPException(status_code=400, detail="Invalid input: All fields are required.")
    
    response=None

    if(request.type=="chatbot"):
        answer = get_best_match(request.question)
        final_answer = None
        if(answer!=None):
            final_answer = calculateFinalMacros(answer, request.question)
        else:
            final_answer = "Ask Queries Related FoodItems or General MacroMinder Related Work Flow."

        response = ChatbotResponse(
            userid=request.userid,
            type=request.type,
            answer=final_answer,
            timestamp=datetime.utcnow()
        )
    else:
        answer = get_best_match(request.question)
        final_answer = answer
        if(answer!=None):
            final_answer = calculateFinalMacrosInfo(answer, request.question)

        response = ChatbotResponse(
            userid=request.userid,
            type=request.type,
            answer=final_answer,
            timestamp=datetime.utcnow()
        )
        
    return response

@chatbot_router.post("/newFoodData")
def add_new_data_to_VectorDB(new_food_item: NewFoodItemRequest):
    try:
        success = add_new_food_item_to_excel(new_food_item)
        return True
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@chatbot_router.post("/newGeneralData")
def add_new_data_to_VectorDB(new_general_item: NewGeneralItemRequest):
    try:
        success = add_new_general_item_to_excel(new_general_item)
        return True
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@chatbot_router.get("/memory-usage")
def total_memory_usage():
    total_memory_mb = 0
    for p in psutil.process_iter(attrs=['pid', 'name']):
        try:
            if 'python' in p.info['name'].lower():  # Check all running Python processes
                total_memory_mb += psutil.Process(p.info['pid']).memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return {"total_memory_mb": total_memory_mb}

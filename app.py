from transformers import pipeline
from fastapi import FastAPI
from controllers.chatbot_controller import chatbot_router
from service.createWordEmbeddings import load_faiss_index

chatBotApp = FastAPI(title="Chatbot Service", description="A chatbot service using Hugging Face and FastAPI.")

# Include routers
chatBotApp.include_router(chatbot_router)

load_faiss_index()

@chatBotApp.get("/")
def home():
    return {"message": "Chatbot Service is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(chatBotApp, host="127.0.0.1", port=5566)

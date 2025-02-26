import os
import signal
import time
import pandas as pd
from model.newFoodItem_model import NewFoodItemRequest
from model.newGeneralInfo_model import NewGeneralItemRequest
from service.createWordEmbeddings import store_embeddings, load_faiss_index
import sys

def add_new_food_item_to_excel(new_food_item: NewFoodItemRequest):
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/MacroMinderFoodItems.xlsx"))

    if not os.path.exists(file_path):
        raise FileNotFoundError("Excel file not found!")

    df = pd.read_excel(file_path)
    new_row = pd.DataFrame([new_food_item.dict()])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(file_path, index=False)

    store_embeddings()  # Update FAISS embeddings
    time.sleep(2)  # Ensure the index file is properly saved

    load_faiss_index()  # Reload FAISS index with latest embeddings

    restart_server()

    return True

def add_new_general_item_to_excel(new_general_item: NewGeneralItemRequest):
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/MacroMinderGeneralItems.xlsx"))

    if not os.path.exists(file_path):
        raise FileNotFoundError("Excel file not found!")

    df = pd.read_excel(file_path)
    new_row = pd.DataFrame([new_general_item.dict()])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(file_path, index=False)

    store_embeddings()  # Update FAISS embeddings
    time.sleep(2)  # Ensure the index file is properly saved

    load_faiss_index()  # Reload FAISS index with latest embeddings

    restart_server()

    return True

def restart_server():
    """Restart server based on environment (local or Docker)."""
    print("Restarting server...")

    if os.environ.get("DOCKERIZED"):  # If running inside Docker
        os.kill(os.getpid(), signal.SIGTERM)  # Stop the process (Docker will restart)
    else:  # If running manually with python app.py
        python = sys.executable
        args = sys.argv
        time.sleep(2)
        os.execv(python, [python] + args)

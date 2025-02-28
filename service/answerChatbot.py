import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pickle
import os
import re
from MLModel.modelLoader import get_model

model = get_model()
INDEX_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../MLModel/vector_index.faiss"))
DATA_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../MLModel/vector_data.pkl"))

index = faiss.read_index(INDEX_FILE)
with open(DATA_FILE, "rb") as f:
    word_embeddings = pickle.load(f)
SIMILARITY_THRESHOLD = 0.3

def get_best_match(question: str) -> str:
    query_embedding = np.array(model.encode([question]), dtype=np.float16)
    _, index_result = index.search(query_embedding, 1)
    best_match_index = index_result[0][0]

    closest_match = word_embeddings[best_match_index]
    best_match_embedding = np.array(model.encode([closest_match]), dtype=np.float16)
    similarity_score = cosine_similarity(query_embedding, best_match_embedding)[0][0]

    if similarity_score < SIMILARITY_THRESHOLD:
        return None
    return closest_match

def extract_quantity(text):
    match = re.search(r"(\d+(?:\.\d+)?)", text)  # Supports integers & decimals
    return float(match.group(1)) if match else None  # Convert to float

def calculateFinalMacros(answer, question):
    requested_quantity = extract_quantity(question)
    if requested_quantity is None:
        return answer  
    
    base_quantity = extract_quantity(answer)
    if base_quantity is None or base_quantity == 0:
        return answer  # Avoid division by zero
    
    # Compute scaling factor
    factor = requested_quantity / base_quantity

    print(f"Base Quantity: {base_quantity}")
    print(f"Requested Quantity: {requested_quantity}")
    print(f"Original Answer: {answer}")

    # Modify macros based on factor
    updated_answer = re.sub(r"(\d+(?:\.\d+)?) grams protein", lambda m: f"{round(float(m.group(1)) * factor, 1)} grams protein", answer)
    updated_answer = re.sub(r"(\d+(?:\.\d+)?) grams carbohydrates", lambda m: f"{round(float(m.group(1)) * factor, 1)} grams carbohydrates", updated_answer)
    updated_answer = re.sub(r"(\d+(?:\.\d+)?) grams fats", lambda m: f"{round(float(m.group(1)) * factor, 1)} grams fats", updated_answer)
    updated_answer = re.sub(r"(\d+(?:\.\d+)?) calories", lambda m: f"{round(float(m.group(1)) * factor, 1)} calories", updated_answer)

    print(f"After Macro Scaling: {updated_answer}")
    
    # Replace base quantity with requested quantity, supporting decimals
    updated_answer = re.sub(rf"\b{base_quantity:g}\s*(gms|ml|unit)\b", f"{requested_quantity:g} \\1", updated_answer)

    print(f"Final Answer: {updated_answer}")

    return updated_answer


def extract_unit(text):
    match = re.search(r"(gms|ml|unit)", text)
    return match.group(1) if match else None

def calculateFinalMacrosInfo(answer, question):
   
    base_quantity = extract_quantity(answer)
    if base_quantity is None or base_quantity == 0:
        return None  # Avoid division by zero
    
    requested_quantity = extract_quantity(question)
    if requested_quantity is None:
        requested_quantity = base_quantity  # If no quantity is found in the question, return None
    
    
    unit = extract_unit(answer)  # Extract unit from answer
    factor = requested_quantity / base_quantity  # Compute scaling factor
    
    # Extract and scale macros
    protein = re.search(r"(\d+(?:\.\d+)?) grams protein", answer)
    carbohydrates = re.search(r"(\d+(?:\.\d+)?) grams carbohydrates", answer)
    fats = re.search(r"(\d+(?:\.\d+)?) grams fats", answer)
    calories = re.search(r"(\d+(?:\.\d+)?) calories", answer)
    
    return {
        "quantity": requested_quantity,
        "unit": unit if unit else None,
        "protein": round(float(protein.group(1)) * factor, 1) if protein else None,
        "carbohydrates": round(float(carbohydrates.group(1)) * factor, 1) if carbohydrates else None,
        "fats": round(float(fats.group(1)) * factor, 1) if fats else None,
        "calories": round(float(calories.group(1)) * factor, 1) if calories else None
    }

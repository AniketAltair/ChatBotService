import faiss
import pandas as pd
import numpy as np
import time
import os
import pickle
from MLModel.modelLoader import get_model

model = get_model()
INDEX_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../MLModel/vector_index.faiss"))
DATA_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../MLModel/vector_data.pkl"))

index = None  # Declare global index
word_embeddings = []  # Store word embeddings in memory


def load_food_data():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/MacroMinderFoodItems.xlsx"))

    if not os.path.exists(file_path):
        print("Error: Excel file not found!")
        return None, None

    df = pd.read_excel(file_path)

    WordEmbeddings = [
        f"{'100' if row['QuantityType'] in ['gms', 'ml'] else '1'} {row['QuantityType']} {row['Name']} has {row['Protein']} grams protein, {row['Carbohydrates']} grams carbohydrates, {row['Fats']} grams fats and {row['Calories']} calories."
        for _, row in df.iterrows()
    ]

    return df, WordEmbeddings

def load_general_data():
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets/MacroMinderGeneralItems.xlsx"))

    if not os.path.exists(file_path):
        print("Error: Excel file not found!")
        return None, None

    df_general = pd.read_excel(file_path)

    WordEmbeddings_general = [
        f"{row['Description']}"
        for _, row in df_general.iterrows()
    ]

    return df_general, WordEmbeddings_general


def store_embeddings():
    global index, word_embeddings  

    df, word_embeddings = load_food_data()
    if df is None:
        return
    
    df_general,WordEmbeddings_general = load_general_data()
    if df_general is None:
        return

    all_embeddings = word_embeddings + WordEmbeddings_general
    vectors = np.array(model.encode(all_embeddings), dtype=np.float16)

    with open(DATA_FILE, "wb") as f:
        pickle.dump(all_embeddings, f)

    d = vectors.shape[1]
    
    # IMPORTANT: Clear the FAISS index before re-adding vectors
    index = faiss.IndexFlatL2(d)  
    index.reset()  # Clears any old embeddings stored in FAISS
    
    index.add(vectors)

    faiss.write_index(index, INDEX_FILE)  # Save the updated FAISS index

    print("Embeddings updated and reloaded into memory!")



def load_faiss_index():
    global index

    if os.path.exists(INDEX_FILE):
        time.sleep(1)  # Small delay to ensure FAISS file is written

        index = faiss.read_index(INDEX_FILE)  
        print("✅ FAISS index reloaded with latest embeddings!")
    else:
        print("FAISS index not found, initializing new index.")
        index = None


# Load the FAISS index on startup
store_embeddings()
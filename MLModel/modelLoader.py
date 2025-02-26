from sentence_transformers import SentenceTransformer

_model = None  # Store model globally

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            trust_remote_code=True
        ).half()  # Convert to float16 (or use `.int8()` for even lower memory)
    return _model

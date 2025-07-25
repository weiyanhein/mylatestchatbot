# # backend/config.py

# import os
# from dotenv import load_dotenv

# load_dotenv() # Load environment variables from .env file

# class Config:
#     # --- Project Paths ---
#     PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     DATA_DIR = os.path.join(PROJECT_ROOT, "data")
#     PRODUCT_DATA_PATH = os.path.join(DATA_DIR, "skincare_products.json")
#     KNOWLEDGE_BASE_DIR = os.path.join(DATA_DIR, "pdfs")
#     UPLOAD_IMAGE_DIR = os.path.join(PROJECT_ROOT, "backend", "uploaded_images")
#     STATIC_PRODUCT_IMAGES_DIR = os.path.join(PROJECT_ROOT, "backend", "static", "product_images")
#     CHROMA_PERSIST_DIRECTORY = os.path.join(DATA_DIR, "chroma_db")
#     TFIDF_MODEL_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
#     INTENT_LABELS_PATH = os.path.join(DATA_DIR, "intent_labels.json")
#     INTENT_TRAINING_VECTORS_PATH = os.path.join(DATA_DIR, "intent_training_vectors.npy")

#     # --- LLM and Agent Configuration ---
#     OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3:8b")
#     OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large:335m")

#     MAX_RECOMMENDED_PRODUCTS = 3
#     INTENT_CONFIDENCE_THRESHOLD = 0.5 # For TF-IDF intent classification

#     # --- Database Configuration ---
#     MONGO_DB_CONNECTION_STRING = os.getenv("MONGO_DB_CONNECTION_STRING", "mongodb://localhost:27017/")
#     MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "skincare_chatbot_db")
#     MONGO_PRODUCTS_COLLECTION = "products"

#     # --- Image Upload Settings ---
#     MAX_IMAGE_SIZE_MB = 5
#     ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
#     INTENT_TRAINING_DATA_PATH: str = os.path.join(DATA_DIR, "intent_training_data.json")
#     INTENT_CLASSIFIER_MODEL_PATH: str = os.path.join(DATA_DIR, "intent_classifier_model.pkl")

#     CHUNK_SIZE = 1000
#     CHUNK_OVERLAP = 200

#     # --- Logging Configuration ---
#     LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # INFO, DEBUG, WARNING, ERROR, CRITICAL

#     # --- Ensure directories exist ---
#     @classmethod
#     def ensure_dirs_exist(cls):
#         os.makedirs(cls.DATA_DIR, exist_ok=True)
#         os.makedirs(cls.KNOWLEDGE_BASE_DIR, exist_ok=True)
#         os.makedirs(cls.UPLOAD_IMAGE_DIR, exist_ok=True)
#         os.makedirs(cls.STATIC_PRODUCT_IMAGES_DIR, exist_ok=True)
#         os.makedirs(cls.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
#         os.makedirs(cls.CHROMA_PERSIST_DIRECTORY + "_kb", exist_ok=True) # Ensure KB persist dir exists
#         os.makedirs(os.path.dirname(cls.TFIDF_MODEL_PATH), exist_ok=True)
#         os.makedirs(os.path.dirname(cls.INTENT_LABELS_PATH), exist_ok=True)
#         os.makedirs(os.path.dirname(cls.INTENT_CLASSIFIER_MODEL_PATH), exist_ok=True)


# # Ensure directories are created when Config is imported
# Config.ensure_dirs_exist()

# # Print configuration paths for debugging during startup (this will be replaced by logger output in app.py)
# print("--- CONFIGURATION PATHS ---")
# print(f"PROJECT_ROOT: {Config.PROJECT_ROOT}")
# print(f"DATA_DIR: {Config.DATA_DIR}")
# print(f"PRODUCT_DATA_PATH: {Config.PRODUCT_DATA_PATH}")
# print(f"KNOWLEDGE_BASE_DIR (PDFs): {Config.KNOWLEDGE_BASE_DIR}")
# print(f"UPLOAD_IMAGE_DIR: {Config.UPLOAD_IMAGE_DIR}")
# print(f"STATIC_PRODUCT_IMAGES_DIR: {Config.STATIC_PRODUCT_IMAGES_DIR}")
# print(f"CHROMA_PERSIST_DIRECTORY: {Config.CHROMA_PERSIST_DIRECTORY}")
# print(f"TFIDF_MODEL_PATH: {Config.TFIDF_MODEL_PATH}")
# print(f"INTENT_LABELS_PATH: {Config.INTENT_LABELS_PATH}")
# print(f"INTENT_TRAINING_VECTORS_PATH: {Config.INTENT_TRAINING_VECTORS_PATH}")
# print(f"OLLAMA_CHAT_MODEL: {Config.OLLAMA_CHAT_MODEL}")
# print(f"OLLAMA_EMBEDDING_MODEL: {Config.OLLAMA_EMBEDDING_MODEL}")
# print(f"LOG_LEVEL: {Config.LOG_LEVEL}")
# # print("--- END CONFIGURATION PATHS ---")

# backend/config.py
# backend/config.py
# backend/config.py

import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class Config:
    # --- Project Paths ---
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    PRODUCT_DATA_PATH = os.path.join(DATA_DIR, "skincare_products.json")
    KNOWLEDGE_BASE_DIR = os.path.join(DATA_DIR, "pdfs")
    UPLOAD_IMAGE_DIR = os.path.join(PROJECT_ROOT, "backend", "uploaded_images")
    STATIC_PRODUCT_IMAGES_DIR = os.path.join(PROJECT_ROOT, "backend", "static", "product_images")
    CHROMA_PERSIST_DIRECTORY = os.path.join(DATA_DIR, "chroma_db")
    TFIDF_MODEL_PATH = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
    INTENT_LABELS_PATH = os.path.join(DATA_DIR, "intent_labels.json")
    INTENT_TRAINING_VECTORS_PATH = os.path.join(DATA_DIR, "intent_training_vectors.npy") # Not currently used, but path exists.

    # --- LLM and Agent Configuration ---
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3:8b")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large:335m")

    # Chunking for PDF ingestion
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    MAX_RECOMMENDED_PRODUCTS = 2
    INTENT_CONFIDENCE_THRESHOLD = 0.3 # For TF-IDF intent classification

    # --- Database Configuration ---
    MONGO_DB_CONNECTION_STRING = os.getenv("MONGO_DB_CONNECTION_STRING", "mongodb://localhost:27017/")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "skincare_chatbot_db")
    MONGO_PRODUCTS_COLLECTION = "products"

    # --- Image Upload Settings ---
    MAX_IMAGE_SIZE_MB = 5
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
    INTENT_TRAINING_DATA_PATH: str = os.path.join(DATA_DIR, "intent_training_data.json")
    INTENT_CLASSIFIER_MODEL_PATH: str = os.path.join(DATA_DIR, "intent_classifier_model.pkl")

    # --- Logging Configuration ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # INFO, DEBUG, WARNING, ERROR, CRITICAL

    # --- Ensure directories exist ---
    @classmethod
    def ensure_dirs_exist(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.KNOWLEDGE_BASE_DIR, exist_ok=True)
        os.makedirs(cls.UPLOAD_IMAGE_DIR, exist_ok=True)
        os.makedirs(cls.STATIC_PRODUCT_IMAGES_DIR, exist_ok=True)
        os.makedirs(cls.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        os.makedirs(cls.CHROMA_PERSIST_DIRECTORY + "_kb", exist_ok=True) # Ensure KB persist dir exists
        
        # Ensure parent directories for model files exist
        os.makedirs(os.path.dirname(cls.TFIDF_MODEL_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(cls.INTENT_LABELS_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(cls.INTENT_CLASSIFIER_MODEL_PATH), exist_ok=True)


# Ensure directories are created when Config is imported
Config.ensure_dirs_exist()

# Print configuration paths for debugging during startup (this will be replaced by logger output in app.py)
print("--- CONFIGURATION PATHS ---")
print(f"PROJECT_ROOT: {Config.PROJECT_ROOT}")
print(f"DATA_DIR: {Config.DATA_DIR}")
print(f"PRODUCT_DATA_PATH: {Config.PRODUCT_DATA_PATH}")
print(f"KNOWLEDGE_BASE_DIR (PDFs): {Config.KNOWLEDGE_BASE_DIR}")
print(f"UPLOAD_IMAGE_DIR: {Config.UPLOAD_IMAGE_DIR}")
print(f"STATIC_PRODUCT_IMAGES_DIR: {Config.STATIC_PRODUCT_IMAGES_DIR}")
print(f"CHROMA_PERSIST_DIRECTORY: {Config.CHROMA_PERSIST_DIRECTORY}")
print(f"TFIDF_MODEL_PATH: {Config.TFIDF_MODEL_PATH}")
print(f"INTENT_LABELS_PATH: {Config.INTENT_LABELS_PATH}")
print(f"INTENT_TRAINING_VECTORS_PATH: {Config.INTENT_TRAINING_VECTORS_PATH}")
print(f"OLLAMA_CHAT_MODEL: {Config.OLLAMA_CHAT_MODEL}")
print(f"OLLAMA_EMBEDDING_MODEL: {Config.OLLAMA_EMBEDDING_MODEL}")
print(f"LOG_LEVEL: {Config.LOG_LEVEL}")
print("--- END CONFIGURATION PATHS ---")
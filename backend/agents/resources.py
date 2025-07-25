# backend/agents/resources.py

import os
# import pickle # REMOVED: No longer needed for loading models
import json
import logging # This import is technically redundant due to logger_config, but harmless.
from typing import Optional
from pymongo import MongoClient
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama # CORRECTED: Use ChatOllama for chat models
import joblib # ADDED/KEPT: Essential for loading joblib-saved models

from backend.config import Config
from backend.logger_config import logger # Import the centralized logger

class ChatbotResources:
    """
    Manages and provides access to all external resources required by the chatbot agents.
    This includes LLM instances, vector stores, database connections, and ML models.
    """
    def __init__(self):
        self.ollama_chat_llm: Optional[ChatOllama] = None # Corrected type hint
        self.ollama_json_llm: Optional[ChatOllama] = None # Corrected type hint
        self.ollama_embedding_model: Optional[OllamaEmbeddings] = None
        self.products_vectorstore: Optional[Chroma] = None
        self.knowledge_base_vectorstore: Optional[Chroma] = None
        self.mongo_client: Optional[MongoClient] = None
        self.products_collection = None # MongoDB collection for products
        self.tfidf_vectorizer = None
        self.intent_classifier_model = None
        self.intent_labels = None
        self.chatbot_app = None # To hold the compiled LangGraph app

    async def initialize_resources(self):
        """
        Initializes all resources. This method should be called once at application startup.
        """
        logger.info("Initializing chatbot resources...")
        try:
            # Initialize Ollama LLM for general chat (no JSON format enforcement)
            # Use ChatOllama for proper chat model interaction
            self.ollama_chat_llm = ChatOllama(model=Config.OLLAMA_CHAT_MODEL, temperature=0.6) # Slightly lower temp for more consistent answers
            logger.info(f"Ollama chat LLM initialized with model: {Config.OLLAMA_CHAT_MODEL}")

            # Initialize Ollama LLM for structured JSON output
            # This is key for product recommendation JSON responses
            self.ollama_json_llm = ChatOllama(model=Config.OLLAMA_CHAT_MODEL, format='json', temperature=0.7)
            logger.info(f"Ollama JSON LLM initialized with model: {Config.OLLAMA_CHAT_MODEL} and format='json'")

            # Initialize Ollama Embeddings
            self.ollama_embedding_model = OllamaEmbeddings(model=Config.OLLAMA_EMBEDDING_MODEL)
            logger.info(f"Ollama embedding model initialized with model: {Config.OLLAMA_EMBEDDING_MODEL}")

            # Initialize MongoDB
            self.mongo_client = MongoClient(Config.MONGO_DB_CONNECTION_STRING)
            self.products_collection = self.mongo_client[Config.MONGO_DB_NAME][Config.MONGO_PRODUCTS_COLLECTION]
            self.mongo_client.admin.command('ping') # Test connection
            logger.info(f"MongoDB connected to database '{Config.MONGO_DB_NAME}' and collection '{Config.MONGO_PRODUCTS_COLLECTION}'.")

            # Initialize ChromaDB for product search
            # Ensure the embedding model is provided
            self.products_vectorstore = Chroma(
                persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
                embedding_function=self.ollama_embedding_model
            )
            logger.info(f"ChromaDB for products loaded from: {Config.CHROMA_PERSIST_DIRECTORY}")

            # Initialize ChromaDB for knowledge base
            self.knowledge_base_vectorstore = Chroma(
                persist_directory=Config.CHROMA_PERSIST_DIRECTORY + "_kb", # Separate directory for KB
                embedding_function=self.ollama_embedding_model
            )
            logger.info(f"ChromaDB for knowledge base loaded from: {Config.CHROMA_PERSIST_DIRECTORY}_kb")


            # Load TF-IDF vectorizer and Intent Classifier Model using joblib
            if (os.path.exists(Config.TFIDF_MODEL_PATH) and
                os.path.exists(Config.INTENT_CLASSIFIER_MODEL_PATH) and
                os.path.exists(Config.INTENT_LABELS_PATH)):
                
                logger.info(f"Loading TF-IDF vectorizer from: {Config.TFIDF_MODEL_PATH}")
                # FIX: Use joblib.load for models saved with joblib.dump
                self.tfidf_vectorizer = joblib.load(Config.TFIDF_MODEL_PATH) 
                logger.info("TF-IDF vectorizer loaded successfully.")
                
                logger.info(f"Loading intent classifier model from: {Config.INTENT_CLASSIFIER_MODEL_PATH}")
                # FIX: Use joblib.load for models saved with joblib.dump
                self.intent_classifier_model = joblib.load(Config.INTENT_CLASSIFIER_MODEL_PATH)
                logger.info("Intent classifier model loaded successfully.")

                with open(Config.INTENT_LABELS_PATH, 'r', encoding='utf-8') as f: # Ensure encoding for json
                    self.intent_labels = json.load(f)
                logger.info("Intent labels loaded successfully.")
            else:
                logger.warning(
                    "One or more intent classification model files (TF-IDF vectorizer, classifier, or labels) "
                    "not found. Intent classification will not function correctly. "
                    "Please run 'python prepare_data.py' to generate them."
                )

            logger.info("All chatbot resources initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize chatbot resources: {e}", exc_info=True)
            # It's crucial to raise the exception to prevent the app from starting with missing resources
            raise

    def close_resources(self):
        """
        Closes any open connections, like the MongoDB client.
        """
        logger.info("Closing chatbot resources...")
        if self.mongo_client:
            self.mongo_client.close()
            logger.info("MongoDB client closed.")
        logger.info("Chatbot resources closed.")

# Instantiate resources globally (or manage through FastAPI dependencies)
resources_instance = ChatbotResources()

# Dependency function for FastAPI to get resources
async def get_chatbot_resources():
    """FastAPI dependency to inject initialized chatbot resources."""
    # This check ensures resources are initialized if FastAPI's startup event somehow misses it
    # or in environments where it might be called out of sequence (e.g., during testing).
    if not resources_instance.ollama_chat_llm:
        logger.warning("get_chatbot_resources called before resources_instance was fully initialized. Initializing now.")
        await resources_instance.initialize_resources()
    return resources_instance

# Function to initialize resources specifically for FastAPI startup
async def init_chatbot_resources_for_fastapi_startup():
    """Called by FastAPI startup event to initialize resources."""
    # This check is crucial to prevent re-initialization during hot-reloads in development
    # (e.g., when running with `uvicorn --reload`)
    if not resources_instance.ollama_chat_llm:
        logger.info("FastAPI startup: Initializing global chatbot resources.")
        await resources_instance.initialize_resources()
    else:
        logger.info("FastAPI startup: Chatbot resources already initialized.")
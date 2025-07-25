# # prepare_data.py

# import os
# import json
# import pickle # Used by joblib for dumping models internally
# import sys
# import glob
# import shutil # For removing directories

# from pymongo import MongoClient
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter # Corrected import from langchain.text_splitter
# from langchain_core.documents import Document # Explicitly import Document for clarity when creating docs

# # Import for intent classification
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC
# # from sklearn.model_selection import train_test_split # Not directly used for full training script, kept for reference
# # from sklearn.metrics import classification_report # Optional, if you want to print evaluation
# import joblib # For saving/loading models

# # Add the backend directory to sys.path to allow importing Config and logger
# script_dir = os.path.dirname(os.path.abspath(__file__))
# backend_dir = os.path.join(script_dir, 'backend')
# if backend_dir not in sys.path:
#     sys.path.insert(0, backend_dir) # Add backend/ to path
# project_root = os.path.dirname(script_dir) # Go up one level from backend to project root
# if project_root not in sys.path:
#     sys.path.insert(0, project_root) # Add project root to path, useful if Config is in root or other top-level imports


# from backend.config import Config
# from backend.logger_config import logger # Import the centralized logger

# # --- Ingestion Functions ---

# def ingest_pdf_data():
#     """Ingests PDF documents from KNOWLEDGE_BASE_DIR into ChromaDB for guidelines."""
#     kb_persist_directory = Config.CHROMA_PERSIST_DIRECTORY + "_kb"
#     logger.info(f"--- Ingesting PDF data into ChromaDB in '{kb_persist_directory}' ---")

#     # Ensure the directory exists, but also consider clearing it for a fresh ingest
#     if os.path.exists(kb_persist_directory):
#         logger.info(f"Removing existing PDF knowledge base ChromaDB at {kb_persist_directory}...")
#         try:
#             shutil.rmtree(kb_persist_directory)
#             logger.info("Existing PDF knowledge base ChromaDB removed.")
#         except OSError as e:
#             logger.error(f"Error removing directory {kb_persist_directory}: {e}", exc_info=True)
#             logger.error("Please ensure you have write permissions and no other process is using the directory.")
#             return # Exit if we can't clean up

#     os.makedirs(kb_persist_directory, exist_ok=True) # Recreate empty directory

#     embedding_function = OllamaEmbeddings(
#         model=Config.OLLAMA_EMBEDDING_MODEL,
#     )
    
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=Config.CHUNK_SIZE,
#         chunk_overlap=Config.CHUNK_OVERLAP
#     )

#     documents = []
#     pdf_files = [f for f in os.listdir(Config.KNOWLEDGE_BASE_DIR) if f.endswith('.pdf')]

#     if not pdf_files:
#         logger.warning(f"No PDF files found in {Config.KNOWLEDGE_BASE_DIR}. Skipping PDF ingestion.")
#         return

#     for pdf_file in pdf_files:
#         pdf_path = os.path.join(Config.KNOWLEDGE_BASE_DIR, pdf_file)
#         logger.info(f"Loading {pdf_path}...")
#         try:
#             loader = PyPDFLoader(pdf_path)
#             docs = loader.load()
#             documents.extend(text_splitter.split_documents(docs))
#             logger.info(f"  - Loaded {len(docs)} pages and split into {len(text_splitter.split_documents(docs))} chunks from {pdf_file}")
#         except Exception as e:
#             logger.error(f"  - ERROR loading {pdf_file}: {e}", exc_info=True)

#     if documents:
#         logger.info(f"Creating/updating vector store for {len(documents)} document chunks (guidelines)...")
#         try:
#             Chroma.from_documents(
#                 documents=documents,
#                 embedding=embedding_function,
#                 persist_directory=kb_persist_directory,
#             )
#             logger.info(f"PDF data ingestion complete for knowledge base in '{kb_persist_directory}'.")
#         except Exception as e:
#             logger.error(f"ERROR creating ChromaDB collection for guidelines: {e}", exc_info=True)
#             logger.error("Please ensure your Ollama embedding model is running and accessible (ollama run " + Config.OLLAMA_EMBEDDING_MODEL + ").")
#     else:
#         logger.warning("No documents to process from PDFs after splitting.")


# def ingest_product_data_into_mongodb():
#     """Ingests product data from JSON file into MongoDB."""
#     logger.info(f"--- Ingesting product data into MongoDB ---")
#     client = None
#     try:
#         client = MongoClient(Config.MONGO_DB_CONNECTION_STRING)
#         db = client[Config.MONGO_DB_NAME]
#         products_collection = db[Config.MONGO_PRODUCTS_COLLECTION]

#         products_collection.delete_many({})
#         logger.info("Cleared existing product data in MongoDB.")

#         if not os.path.exists(Config.PRODUCT_DATA_PATH):
#             logger.error(f"ERROR: Product data file not found at {Config.PRODUCT_DATA_PATH}. Please ensure it exists.")
#             return

#         with open(Config.PRODUCT_DATA_PATH, 'r', encoding='utf-8') as f:
#             products_data = json.load(f)

#         if products_data:
#             products_collection.insert_many(products_data)
#             logger.info(f"Successfully inserted {len(products_data)} products into MongoDB.")
#         else:
#             logger.warning("No product data found in JSON file.")
#     except Exception as e:
#         logger.error(f"ERROR ingesting product data into MongoDB: {e}", exc_info=True)
#     finally:
#         if client:
#             client.close()


# def ingest_product_data_into_chromadb():
#     """Ingests product data from JSON file into ChromaDB for vector search."""
#     logger.info(f"--- Ingesting product data into ChromaDB for vector search ---")
    
#     # Ensure the directory exists, but also consider clearing it for a fresh ingest
#     if os.path.exists(Config.CHROMA_PERSIST_DIRECTORY):
#         logger.info(f"Removing existing product ChromaDB at {Config.CHROMA_PERSIST_DIRECTORY}...")
#         try:
#             shutil.rmtree(Config.CHROMA_PERSIST_DIRECTORY)
#             logger.info("Existing product ChromaDB removed.")
#         except OSError as e:
#             logger.error(f"Error removing directory {Config.CHROMA_PERSIST_DIRECTORY}: {e}", exc_info=True)
#             logger.error("Please ensure you have write permissions and no other process is using the directory.")
#             return # Exit if we can't clean up

#     os.makedirs(Config.CHROMA_PERSIST_DIRECTORY, exist_ok=True) # Recreate empty directory


#     embedding_function = OllamaEmbeddings(
#         model=Config.OLLAMA_EMBEDDING_MODEL,
#     )

#     if not os.path.exists(Config.PRODUCT_DATA_PATH):
#         logger.error(f"ERROR: Product data file not found at {Config.PRODUCT_DATA_PATH}. Cannot ingest into ChromaDB.")
#         return

#     with open(Config.PRODUCT_DATA_PATH, 'r', encoding='utf-8') as f:
#         products_data = json.load(f)

#     if not products_data:
#         logger.warning("No product data found in JSON file. Skipping ChromaDB product ingestion.")
#         return

#     documents_to_embed = []
#     metadatas = []

#     for product in products_data:
#         text_content = (
#             f"Product Name: {product.get('name', 'N/A')}. "
#             f"Brand: {product.get('brand', 'N/A')}. "
#             f"Product Type: {product.get('product_type', 'N/A')}. " # Use product_type
#             f"Ingredients: {', '.join(product.get('ingredients', []))}. "
#             f"Benefits: {', '.join(product.get('benefits', []))}. "
#             f"Concerns: {', '.join(product.get('concerns', []))}. "
#             f"Usage: {product.get('usage', 'N/A')}."
#         )
#         text_content = ' '.join(text_content.split()).strip()

#         if not text_content:
#             logger.warning(f"WARNING: Skipping product {product.get('product_id', 'Unknown ID')} due to empty text content for embedding.")
#             continue

#         # Use Product ID (consistent with MongoDB and models)
#         product_id = product.get("product_id")
#         if product_id is None:
#             logger.warning(f"Product {product.get('name', 'Unknown Name')} has no 'product_id'. Skipping for ChromaDB.")
#             continue

#         documents_to_embed.append(Document(
#             page_content=text_content,
#             metadata={
#                 "product_id": product_id, # Use product_id
#                 "name": product.get("name"),
#                 "brand": product.get("brand"),
#                 "product_type": product.get("product_type"), # Use product_type
#                 "image_filename": product.get("image_filename"),
#                 # "embedded_content_text": text_content # Redundant if page_content is directly embedded
#             }
#         ))

#     if documents_to_embed:
#         logger.info(f"Embedding and ingesting {len(documents_to_embed)} products into ChromaDB...")
#         try:
#             Chroma.from_documents(
#                 documents=documents_to_embed, # Pass list of Document objects
#                 embedding=embedding_function,
#                 persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
#             )
#             logger.info(f"Product data ingestion into ChromaDB complete for '{Config.CHROMA_PERSIST_DIRECTORY}'.")
#         except Exception as e:
#             logger.error(f"ERROR creating ChromaDB collection for products: {e}", exc_info=True)
#             logger.error("Please ensure your Ollama embedding model is running and accessible (ollama run " + Config.OLLAMA_EMBEDDING_MODEL + ").")
#     else:
#         logger.warning("No products available to embed after filtering.")


# def train_and_save_intent_classifier():
#     """
#     Loads intent training data, trains a TF-IDF vectorizer and a LinearSVC classifier,
#     and saves them to disk.
#     """
#     logger.info(f"--- Training and Saving Intent Classifier ---")

#     if not os.path.exists(Config.INTENT_TRAINING_DATA_PATH):
#         logger.error(f"ERROR: Intent training data file not found at {Config.INTENT_TRAINING_DATA_PATH}.")
#         logger.error("Please create this file with your intent training examples.")
#         return

#     try:
#         with open(Config.INTENT_TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
#             training_data = json.load(f)

#         if not training_data:
#             logger.warning("No training data found in JSON file. Skipping intent classifier training.")
#             return

#         texts = [item['text'] for item in training_data]
#         labels = [item['intent'] for item in training_data]

#         if len(set(labels)) < 2:
#             logger.warning("WARNING: Need at least 2 distinct intent classes to train a classifier. Skipping.")
#             return

#         # Initialize TF-IDF Vectorizer
#         logger.info("Fitting TF-IDF Vectorizer...")
#         tfidf_vectorizer = TfidfVectorizer()
#         X = tfidf_vectorizer.fit_transform(texts)
        
#         # Save the TF-IDF Vectorizer
#         joblib.dump(tfidf_vectorizer, Config.TFIDF_MODEL_PATH)
#         logger.info(f"TF-IDF Vectorizer saved to {Config.TFIDF_MODEL_PATH}")

#         # Train LinearSVC Classifier
#         logger.info("Training LinearSVC Classifier...")
        
#         # We need to map labels to integers for scikit-learn
#         unique_labels = sorted(list(set(labels)))
#         label_to_int = {label: i for i, label in enumerate(unique_labels)}
#         int_labels = [label_to_int[label] for label in labels]

#         classifier = LinearSVC(random_state=42, max_iter=2000) # Increased max_iter for convergence
#         classifier.fit(X, int_labels)

#         # Save the trained classifier model
#         joblib.dump(classifier, Config.INTENT_CLASSIFIER_MODEL_PATH)
#         logger.info(f"Intent Classifier model saved to {Config.INTENT_CLASSIFIER_MODEL_PATH}")

#         # Save the intent labels mapping
#         with open(Config.INTENT_LABELS_PATH, 'w', encoding='utf-8') as f:
#             json.dump(unique_labels, f, indent=4) # Save unique_labels directly
#         logger.info(f"Intent Labels saved to {Config.INTENT_LABELS_PATH}")

#         logger.info("Intent classifier training and saving complete.")

#         # Optional: Print a classification report for self-evaluation
#         # This part is commented out as it's typically for development/debugging,
#         # but you can uncomment if you want to see the report when running this script.
#         # if len(set(labels)) > 1 and len(labels) > 0: # Ensure enough data for report
#         #     y_pred = classifier.predict(X)
#         #     int_to_label = {i: label for label, i in label_to_int.items()}
#         #     predicted_labels = [int_to_label[i] for i in y_pred]
#         #     from sklearn.metrics import classification_report
#         #     logger.info("\n--- Intent Classifier Training Report (on training data) ---")
#         #     logger.info(classification_report(labels, predicted_labels, target_names=unique_labels))
#         #     logger.info("----------------------------------------------------------")

#     except Exception as e:
#         logger.error(f"ERROR training and saving intent classifier: {e}", exc_info=True)
#         logger.error("Please ensure your 'intent_training_data.json' is correctly formatted and contains enough data.")


# if __name__ == "__main__":
#     logger.info("--- Starting Data Preparation ---")
    
#     # Ensure necessary directories exist before starting ingestion
#     Config.ensure_dirs_exist()
#     logger.info("All necessary directories ensured.")
    
#     # Run ingestion and training functions
#     # Order matters: MongoDB first for product data, then ChromaDB, then classifier
#     ingest_product_data_into_mongodb() 
#     ingest_product_data_into_chromadb() 
#     ingest_pdf_data() # Knowledge base from PDFs
#     train_and_save_intent_classifier()
    
#     logger.info("\n--- Data Preparation Complete ---")

# prepare_data.py

import os
import json
import pickle # Used by joblib for dumping models internally - keep for clarity if joblib uses it.
import sys
import glob
import shutil # For removing directories

from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter # Corrected import from langchain.text_splitter
from langchain_core.documents import Document # Explicitly import Document for clarity when creating docs

# Import for intent classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression # CHANGED: From LinearSVC to LogisticRegression
# from sklearn.model_selection import train_test_split # Not directly used, kept for reference
# from sklearn.metrics import classification_report # Optional, if you want to print evaluation
import joblib # For saving/loading models

# Add the backend directory to sys.path to allow importing Config and logger
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(script_dir, 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir) # Add backend/ to path
project_root = os.path.dirname(script_dir) # Go up one level from backend to project root
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Add project root to path, useful if Config is in root or other top-level imports


from backend.config import Config
from backend.logger_config import logger # Import the centralized logger

# --- Ingestion Functions ---

def ingest_pdf_data():
    """Ingests PDF documents from KNOWLEDGE_BASE_DIR into ChromaDB for guidelines."""
    kb_persist_directory = Config.CHROMA_PERSIST_DIRECTORY + "_kb"
    logger.info(f"--- Ingesting PDF data into ChromaDB in '{kb_persist_directory}' ---")

    # Ensure the directory exists, but also consider clearing it for a fresh ingest
    if os.path.exists(kb_persist_directory):
        logger.info(f"Removing existing PDF knowledge base ChromaDB at {kb_persist_directory}...")
        try:
            shutil.rmtree(kb_persist_directory)
            logger.info("Existing PDF knowledge base ChromaDB removed.")
        except OSError as e:
            logger.error(f"Error removing directory {kb_persist_directory}: {e}", exc_info=True)
            logger.error("Please ensure you have write permissions and no other process is using the directory.")
            return # Exit if we can't clean up

    os.makedirs(kb_persist_directory, exist_ok=True) # Recreate empty directory

    embedding_function = OllamaEmbeddings(
        model=Config.OLLAMA_EMBEDDING_MODEL,
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )

    documents = []
    pdf_files = [f for f in os.listdir(Config.KNOWLEDGE_BASE_DIR) if f.endswith('.pdf')]

    if not pdf_files:
        logger.warning(f"No PDF files found in {Config.KNOWLEDGE_BASE_DIR}. Skipping PDF ingestion.")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(Config.KNOWLEDGE_BASE_DIR, pdf_file)
        logger.info(f"Loading {pdf_path}...")
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(text_splitter.split_documents(docs))
            logger.info(f"   - Loaded {len(docs)} pages and split into {len(text_splitter.split_documents(docs))} chunks from {pdf_file}")
        except Exception as e:
            logger.error(f"   - ERROR loading {pdf_file}: {e}", exc_info=True)

    if documents:
        logger.info(f"Creating/updating vector store for {len(documents)} document chunks (guidelines)...")
        try:
            Chroma.from_documents(
                documents=documents,
                embedding=embedding_function,
                persist_directory=kb_persist_directory,
            )
            logger.info(f"PDF data ingestion complete for knowledge base in '{kb_persist_directory}'.")
        except Exception as e:
            logger.error(f"ERROR creating ChromaDB collection for guidelines: {e}", exc_info=True)
            logger.error("Please ensure your Ollama embedding model is running and accessible (ollama run " + Config.OLLAMA_EMBEDDING_MODEL + ").")
    else:
        logger.warning("No documents to process from PDFs after splitting.")


def ingest_product_data_into_mongodb():
    """Ingests product data from JSON file into MongoDB."""
    logger.info(f"--- Ingesting product data into MongoDB ---")
    client = None
    try:
        client = MongoClient(Config.MONGO_DB_CONNECTION_STRING)
        db = client[Config.MONGO_DB_NAME]
        products_collection = db[Config.MONGO_PRODUCTS_COLLECTION]

        products_collection.delete_many({})
        logger.info("Cleared existing product data in MongoDB.")

        if not os.path.exists(Config.PRODUCT_DATA_PATH):
            logger.error(f"ERROR: Product data file not found at {Config.PRODUCT_DATA_PATH}. Please ensure it exists.")
            return

        with open(Config.PRODUCT_DATA_PATH, 'r', encoding='utf-8') as f:
            products_data = json.load(f)

        if products_data:
            products_collection.insert_many(products_data)
            logger.info(f"Successfully inserted {len(products_data)} products into MongoDB.")
        else:
            logger.warning("No product data found in JSON file.")
    except Exception as e:
        logger.error(f"ERROR ingesting product data into MongoDB: {e}", exc_info=True)
    finally:
        if client:
            client.close()


def ingest_product_data_into_chromadb():
    """Ingests product data from JSON file into ChromaDB for vector search."""
    logger.info(f"--- Ingesting product data into ChromaDB for vector search ---")
    
    # Ensure the directory exists, but also consider clearing it for a fresh ingest
    if os.path.exists(Config.CHROMA_PERSIST_DIRECTORY):
        logger.info(f"Removing existing product ChromaDB at {Config.CHROMA_PERSIST_DIRECTORY}...")
        try:
            shutil.rmtree(Config.CHROMA_PERSIST_DIRECTORY)
            logger.info("Existing product ChromaDB removed.")
        except OSError as e:
            logger.error(f"Error removing directory {Config.CHROMA_PERSIST_DIRECTORY}: {e}", exc_info=True)
            logger.error("Please ensure you have write permissions and no other process is using the directory.")
            return # Exit if we can't clean up

    os.makedirs(Config.CHROMA_PERSIST_DIRECTORY, exist_ok=True) # Recreate empty directory


    embedding_function = OllamaEmbeddings(
        model=Config.OLLAMA_EMBEDDING_MODEL,
    )

    if not os.path.exists(Config.PRODUCT_DATA_PATH):
        logger.error(f"ERROR: Product data file not found at {Config.PRODUCT_DATA_PATH}. Cannot ingest into ChromaDB.")
        return

    with open(Config.PRODUCT_DATA_PATH, 'r', encoding='utf-8') as f:
        products_data = json.load(f)

    if not products_data:
        logger.warning("No product data found in JSON file. Skipping ChromaDB product ingestion.")
        return

    documents_to_embed = []
    metadatas = []

    for product in products_data:
        text_content = (
            f"Product Name: {product.get('name', 'N/A')}. "
            f"Brand: {product.get('brand', 'N/A')}. "
            f"Product Type: {product.get('product_type', 'N/A')}. " # Use product_type
            f"Ingredients: {', '.join(product.get('ingredients', []))}. "
            f"Benefits: {', '.join(product.get('benefits', []))}. "
            f"Concerns: {', '.join(product.get('concerns', []))}. "
            f"Usage: {product.get('usage', 'N/A')}."
        )
        text_content = ' '.join(text_content.split()).strip()

        if not text_content:
            logger.warning(f"WARNING: Skipping product {product.get('product_id', 'Unknown ID')} due to empty text content for embedding.")
            continue

        # Use Product ID (consistent with MongoDB and models)
        product_id = product.get("product_id")
        if product_id is None:
            logger.warning(f"Product {product.get('name', 'Unknown Name')} has no 'product_id'. Skipping for ChromaDB.")
            continue

        documents_to_embed.append(Document(
            page_content=text_content,
            metadata={
                "product_id": product_id, # Use product_id
                "name": product.get("name"),
                "brand": product.get("brand"),
                "product_type": product.get("product_type"), # Use product_type
                "image_filename": product.get("image_filename"),
                # "embedded_content_text": text_content # Redundant if page_content is directly embedded
            }
        ))

    if documents_to_embed:
        logger.info(f"Embedding and ingesting {len(documents_to_embed)} products into ChromaDB...")
        try:
            Chroma.from_documents(
                documents=documents_to_embed, # Pass list of Document objects
                embedding=embedding_function,
                persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
            )
            logger.info(f"Product data ingestion into ChromaDB complete for '{Config.CHROMA_PERSIST_DIRECTORY}'.")
        except Exception as e:
            logger.error(f"ERROR creating ChromaDB collection for products: {e}", exc_info=True)
            logger.error("Please ensure your Ollama embedding model is running and accessible (ollama run " + Config.OLLAMA_EMBEDDING_MODEL + ").")
    else:
        logger.warning("No products available to embed after filtering.")


def train_and_save_intent_classifier():
    """
    Loads intent training data, trains a TF-IDF vectorizer and a LogisticRegression classifier,
    and saves them to disk.
    """
    logger.info(f"--- Training and Saving Intent Classifier ---")

    if not os.path.exists(Config.INTENT_TRAINING_DATA_PATH):
        logger.error(f"ERROR: Intent training data file not found at {Config.INTENT_TRAINING_DATA_PATH}.")
        logger.error("Please create this file with your intent training examples.")
        return

    try:
        with open(Config.INTENT_TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
            training_data = json.load(f)

        if not training_data:
            logger.warning("No training data found in JSON file. Skipping intent classifier training.")
            return

        texts = [item['text'] for item in training_data]
        labels = [item['intent'] for item in training_data]

        if len(set(labels)) < 2:
            logger.warning("WARNING: Need at least 2 distinct intent classes to train a classifier. Skipping.")
            return

        # Initialize TF-IDF Vectorizer
        logger.info("Fitting TF-IDF Vectorizer...")
        tfidf_vectorizer = TfidfVectorizer()
        X = tfidf_vectorizer.fit_transform(texts)
        
        # Save the TF-IDF Vectorizer
        joblib.dump(tfidf_vectorizer, Config.TFIDF_MODEL_PATH)
        logger.info(f"TF-IDF Vectorizer saved to {Config.TFIDF_MODEL_PATH}")

        # Train Classifier
        logger.info("Training LogisticRegression Classifier...") # CHANGED: Log message
        
        # We need to map labels to integers for scikit-learn
        unique_labels = sorted(list(set(labels)))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        int_labels = [label_to_int[label] for label in labels]

        # CHANGED: From LinearSVC to LogisticRegression
        classifier = LogisticRegression(random_state=42, max_iter=2000, solver='liblinear', multi_class='ovr') 
        
        classifier.fit(X, int_labels)

        # Save the trained classifier model
        joblib.dump(classifier, Config.INTENT_CLASSIFIER_MODEL_PATH)
        logger.info(f"Intent Classifier model saved to {Config.INTENT_CLASSIFIER_MODEL_PATH}")

        # Save the intent labels mapping
        with open(Config.INTENT_LABELS_PATH, 'w', encoding='utf-8') as f:
            json.dump(unique_labels, f, indent=4) # Save unique_labels directly
        logger.info(f"Intent Labels saved to {Config.INTENT_LABELS_PATH}")

        logger.info("Intent classifier training and saving complete.")

        # Optional: Print a classification report for self-evaluation
        # This part is commented out as it's typically for development/debugging,
        # but you can uncomment if you want to see the report when running this script.
        # if len(set(labels)) > 1 and len(labels) > 0: # Ensure enough data for report
        #     y_pred = classifier.predict(X)
        #     int_to_label = {i: label for label, i in label_to_int.items()}
        #     predicted_labels = [int_to_label[i] for i in y_pred]
        #     from sklearn.metrics import classification_report
        #     logger.info("\n--- Intent Classifier Training Report (on training data) ---")
        #     logger.info(classification_report(labels, predicted_labels, target_names=unique_labels))
        #     logger.info("----------------------------------------------------------")

    except Exception as e:
        logger.error(f"ERROR training and saving intent classifier: {e}", exc_info=True)
        logger.error("Please ensure your 'intent_training_data.json' is correctly formatted and contains enough data.")


if __name__ == "__main__":
    logger.info("--- Starting Data Preparation ---")
    
    # Ensure necessary directories exist before starting ingestion
    Config.ensure_dirs_exist()
    logger.info("All necessary directories ensured.")
    
    # Run ingestion and training functions
    # Order matters: MongoDB first for product data, then ChromaDB, then classifier
    ingest_product_data_into_mongodb() 
    ingest_product_data_into_chromadb() 
    ingest_pdf_data() # Knowledge base from PDFs
    train_and_save_intent_classifier()
    
    logger.info("\n--- Data Preparation Complete ---")
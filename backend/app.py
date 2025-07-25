# # backend/app.py

# import os
# import shutil
# import uuid
# import logging # This import is effectively overridden by logger_config, but good practice.
# from typing import Dict, Any, List, Optional
# from contextlib import asynccontextmanager

# from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Request
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles

# from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
# from langgraph.graph import Graph

# # Import custom modules
# from backend.config import Config
# from backend.logger_config import logger # Import the centralized logger
# from backend.models import ChatbotState, ChatRequest, ChatResponse, SkinAnalysisResult, RecommendedProductForAPI
# from backend.agents.resources import get_chatbot_resources, init_chatbot_resources_for_fastapi_startup, resources_instance, ChatbotResources
# from backend.agents.general_conversation import general_conversation_agent
# from backend.agents.product_recommender import product_recommendation_agent
# from backend.agents.knowledge_base_query import query_knowledge_base_agent
# from backend.agents.intent_classifier import classify_intent_agent
# from backend.agents.skin_analyzer import skin_analysis_agent
# from backend.agents.product_comparison import product_comparison_agent # Placeholder agent
# from backend.agents import tools # Import tools module to access get_skin_type_from_image

# # --- FastAPI Application Setup ---

# # Define an async context manager for managing application lifecycle events (startup/shutdown)
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Handles startup and shutdown events for the FastAPI application.
#     Initializes and cleans up chatbot resources.
#     """
#     logger.info("FastAPI app starting up...")
#     # Initialize resources
#     await init_chatbot_resources_for_fastapi_startup()
#     logger.info("Chatbot resources initialized.")

#     # Attach the compiled graph to the app instance for easy access in routes
#     app.chatbot_graph = create_langgraph_workflow()
#     logger.info("LangGraph workflow compiled.")

#     # Yield control to the application
#     yield

#     # --- Shutdown events ---
#     logger.info("FastAPI app shutting down...")
#     # Close resources
#     resources_instance.close_resources()
#     logger.info("Chatbot resources closed.")
#     logger.info("FastAPI app shut down.")

# app = FastAPI(
#     title="Skincare Chatbot API",
#     description="An AI-powered chatbot for skincare recommendations, advice, and skin analysis.",
#     version="1.0.0",
#     lifespan=lifespan # Attach the lifespan context manager
# )

# # --- Serve Static Product Images ---
# # This makes images in backend/static/product_images accessible via /static/product_images/filename.jpg
# try:
#     app.mount("/static/product_images", StaticFiles(directory=Config.STATIC_PRODUCT_IMAGES_DIR), name="static_product_images")
#     logger.info(f"Serving static product images from '{Config.STATIC_PRODUCT_IMAGES_DIR}' at '/static/product_images'")
# except Exception as e:
#     logger.error(f"Failed to mount static files directory '{Config.STATIC_PRODUCT_IMAGES_DIR}': {e}", exc_info=True)


# # --- LangGraph Workflow Definition ---
# # This function creates and compiles the LangGraph once during startup
# def create_langgraph_workflow():
#     """
#     Defines and compiles the LangGraph workflow for the chatbot.
#     """
#     logger.info("Creating LangGraph workflow...")

#     # Define the graph state
#     graph_state = ChatbotState

#     # Create the graph
#     workflow = Graph(graph_state)

#     # Add nodes for each agent. Each node executes an agent function.
#     # We pass 'resources' as a partial function argument to each agent.
#     workflow.add_node("classify_intent_node", lambda state: classify_intent_agent(state, resources_instance))
#     workflow.add_node("general_conversation_node", lambda state: general_conversation_agent(state, resources_instance))
#     workflow.add_node("product_recommendation_node", lambda state: product_recommendation_agent(state, resources_instance))
#     workflow.add_node("query_knowledge_base_node", lambda state: query_knowledge_base_agent(state, resources_instance))
#     workflow.add_node("skin_analysis_node", lambda state: skin_analysis_agent(state, resources_instance))
#     workflow.add_node("product_comparison_node", lambda state: product_comparison_agent(state, resources_instance))


#     # Set the entry point for the graph
#     workflow.set_entry_point("classify_intent_node")

#     # Define conditional edges based on intent classification
#     workflow.add_conditional_edges(
#         "classify_intent_node",
#         # This function determines the next node based on the 'next_node' key set by the intent classifier
#         lambda state: state['next_node'],
#         {
#             "general_conversation_agent": "general_conversation_node",
#             "product_recommendation_agent": "product_recommendation_node",
#             "query_knowledge_base_agent": "query_knowledge_base_node",
#             "skin_analysis_agent": "skin_analysis_node",
#             "product_comparison_agent": "product_comparison_node",
#             # Add other intents as they are defined and mapped
#         }
#     )

#     # Define direct edges from each agent node back to the END, as they produce a final response
#     workflow.add_edge("general_conversation_node", "END")
#     workflow.add_edge("product_recommendation_node", "END")
#     workflow.add_edge("query_knowledge_base_node", "END")
#     workflow.add_edge("skin_analysis_node", "END")
#     workflow.add_edge("product_comparison_node", "END")

#     compiled_graph = workflow.compile()
#     logger.info("LangGraph workflow compiled successfully.")
#     return compiled_graph

# # The graph will be compiled during the FastAPI startup event and attached to app.chatbot_graph


# # --- Helper Function for Chat History Conversion ---
# def convert_lc_history_to_api_history(lc_history: List[BaseMessage]) -> List[Dict[str, str]]:
#     """Converts LangChain BaseMessage history to a simplified dict format for API."""
#     api_history = []
#     for msg in lc_history:
#         if isinstance(msg, HumanMessage):
#             api_history.append({"human": msg.content})
#         elif isinstance(msg, AIMessage):
#             api_history.append({"ai": msg.content})
#     return api_history


# # --- API Endpoints ---

# @app.get("/", summary="Health Check")
# async def root():
#     """
#     Root endpoint for a simple health check.
#     """
#     logger.info("Root endpoint hit - health check.")
#     return {"message": "Skincare Chatbot API is running!"}


# @app.post("/chat", response_model=ChatResponse, summary="Send a chat message to the chatbot")
# async def chat_endpoint(request: ChatRequest, request_fastapi: Request):
#     """
#     Main endpoint for interacting with the skincare chatbot.
#     Receives a user message and optional chat history, processes it through LangGraph,
#     and returns a chatbot response, potentially including product recommendations or skin analysis results.
#     """
#     try:
#         logger.info(f"Received chat request: User message='{request.user_message}'")
#         logger.debug(f"Chat history from request: {request.chat_history}")
#         logger.debug(f"Skin analysis result from request: {request.skin_analysis_result}")

#         # Convert API chat history format to LangChain BaseMessage format for internal use
#         lc_chat_history: List[BaseMessage] = []
#         if request.chat_history:
#             for msg_dict in request.chat_history:
#                 if "human" in msg_dict:
#                     lc_chat_history.append(HumanMessage(content=msg_dict["human"]))
#                 elif "ai" in msg_dict:
#                     lc_chat_history.append(AIMessage(content=msg_dict["ai"]))
        
#         # Initialize the LangGraph state
#         initial_state: ChatbotState = {
#             "user_message": request.user_message,
#             "chat_history": lc_chat_history,
#             "intent": "unclassified", # Will be set by intent classifier
#             "next_node": "", # Will be set by intent classifier
#             "response": "", # Will be set by agents
#             "product_recommendation_data": None,
#             "skin_analysis_result_data": request.skin_analysis_result, # Pass existing analysis data if present
#             "retrieved_docs": None,
#             "comparison_products": None,
#         }

#         logger.debug("Invoking LangGraph with initial state.")
#         # Invoke the LangGraph workflow
#         final_state: ChatbotState = await app.chatbot_graph.ainvoke(initial_state)
#         logger.debug(f"LangGraph execution finished. Final state: {final_state}")

#         # Prepare the response for the API
#         response_text = final_state.get("response", "I'm sorry, I couldn't process your request fully. Please try again.")
        
#         # Update chat history with the current turn for the next request
#         updated_lc_chat_history = lc_chat_history + [
#             HumanMessage(content=request.user_message),
#             AIMessage(content=response_text)
#         ]
#         api_chat_history = convert_lc_history_to_api_history(updated_lc_chat_history)

#         api_product_recommendations = final_state.get("product_recommendation_data")
#         if api_product_recommendations:
#             logger.info(f"Returning {len(api_product_recommendations)} product recommendations.")
        
#         api_skin_analysis_result = final_state.get("skin_analysis_result_data")
#         if api_skin_analysis_result:
#             logger.info("Returning skin analysis result.")

#         logger.info("Chat request processed successfully.")
#         return ChatResponse(
#             response=response_text,
#             chat_history=api_chat_history,
#             product_recommendation=api_product_recommendations,
#             skin_analysis_result=api_skin_analysis_result
#         )

#     except Exception as e:
#         logger.error(f"Error during chat processing: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"An internal error occurred: {e}"
#         )


# @app.post("/upload_image", response_model=SkinAnalysisResult, summary="Upload an image for skin analysis")
# async def upload_image_endpoint(file: UploadFile = File(...)):
#     """
#     Receives an image file, saves it, and performs a mock skin analysis.
#     Returns the skin analysis result.
#     """
#     logger.info(f"Received image upload request for file: {file.filename}, Content-Type: {file.content_type}")

#     # 1. Validate file type
#     if file.content_type not in ["image/jpeg", "image/png"]:
#         logger.warning(f"Invalid file type uploaded: {file.content_type}")
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Only JPEG and PNG images are allowed."
#         )

#     # 2. Generate a unique filename and path
#     file_extension = file.filename.split(".")[-1]
#     unique_filename = f"{uuid.uuid4()}.{file_extension}"
#     upload_path = os.path.join(Config.UPLOAD_IMAGE_DIR, unique_filename)

#     # Ensure the upload directory exists
#     os.makedirs(Config.UPLOAD_IMAGE_DIR, exist_ok=True)

#     # 3. Save the image
#     try:
#         # Save file in chunks to handle large files efficiently
#         with open(upload_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         logger.info(f"Image saved successfully to: {upload_path}")

#         # 4. Perform skin analysis using the tools agent
#         skin_analysis_result = tools.get_skin_type_from_image(upload_path)

#         if skin_analysis_result is None:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Failed to perform skin analysis on the uploaded image."
#             )

#         logger.info(f"Skin analysis completed: Skin Type='{skin_analysis_result.skin_type}'")
#         return skin_analysis_result

#     except HTTPException as e:
#         # Re-raise HTTPExceptions directly
#         raise
#     except Exception as e:
#         logger.error(f"Error during image upload or analysis: {e}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="An error occurred during image processing."
#         )
#     finally:
#         # Optional: Clean up the uploaded file after processing
#         # In a production system, you might move it to a permanent storage or process it asynchronously
#         if os.path.exists(upload_path):
#             try:
#                 os.remove(upload_path)
#                 logger.debug(f"Cleaned up uploaded image file: {upload_path}")
#             except Exception as e:
#                 logger.error(f"Failed to remove uploaded image file '{upload_path}': {e}")


# # Example usage (for local testing with uvicorn):
# # uvicorn backend.app:app --reload --port 8000


# backend/app.py
# backend/app.py

import os
import shutil
import uuid
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# --- CRITICAL FIX HERE ---
# Previous incorrect: from langgraph.graph import Graph
# Correct: Import StateGraph for stateful workflows and END constant
from langgraph.graph import StateGraph, END 
# --- END CRITICAL FIX ---

# Import custom modules
from backend.config import Config
from backend.logger_config import logger # Import the centralized logger
from backend.models import ChatbotState, ChatRequest, ChatResponse, SkinAnalysisResult, RecommendedProductForAPI
from backend.agents.resources import get_chatbot_resources, init_chatbot_resources_for_fastapi_startup, resources_instance, ChatbotResources
from backend.agents.general_conversation import general_conversation_agent
from backend.agents.product_recommender import product_recommendation_agent
from backend.agents.knowledge_base_query import query_knowledge_base_agent
from backend.agents.intent_classifier import classify_intent_agent
from backend.agents.skin_analyzer import skin_analysis_agent
from backend.agents.product_comparison import product_comparison_agent # Placeholder agent
from backend.agents import tools # Import tools module to access get_skin_type_from_image

# --- FastAPI Application Setup ---

# Define an async context manager for managing application lifecycle events (startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Initializes and cleans up chatbot resources.
    """
    logger.info("FastAPI app starting up...")
    # Initialize resources
    await init_chatbot_resources_for_fastapi_startup()
    logger.info("Chatbot resources initialized.")

    # Attach the compiled graph to the app instance for easy access in routes
    app.chatbot_graph = create_langgraph_workflow()
    logger.info("LangGraph workflow compiled.")

    # Yield control to the application
    yield

    # --- Shutdown events ---
    logger.info("FastAPI app shutting down...")
    # Close resources
    resources_instance.close_resources()
    logger.info("Chatbot resources closed.")
    logger.info("FastAPI app shut down.")

app = FastAPI(
    title="Skincare Chatbot API",
    description="An AI-powered chatbot for skincare recommendations, advice, and skin analysis.",
    version="1.0.0",
    lifespan=lifespan # Attach the lifespan context manager
)

# --- Serve Static Product Images ---
# This makes images in backend/static/product_images accessible via /static/product_images/filename.jpg
try:
    app.mount("/static/product_images", StaticFiles(directory=Config.STATIC_PRODUCT_IMAGES_DIR), name="static_product_images")
    logger.info(f"Serving static product images from '{Config.STATIC_PRODUCT_IMAGES_DIR}' at '/static/product_images'")
except Exception as e:
    logger.error(f"Failed to mount static files directory '{Config.STATIC_PRODUCT_IMAGES_DIR}': {e}", exc_info=True)


# --- LangGraph Workflow Definition ---
# This function creates and compiles the LangGraph once during startup
def create_langgraph_workflow():
    """
    Defines and compiles the LangGraph workflow for the chatbot.
    """
    logger.info("Creating LangGraph workflow...")

    # Define the graph state
    graph_state = ChatbotState

    # Create the graph using StateGraph (Correct for stateful workflows)
    workflow = StateGraph(graph_state) 

    # Add nodes for each agent. Each node executes an agent function.
    # We pass 'resources' as a partial function argument to each agent.
    workflow.add_node("classify_intent_node", lambda state: classify_intent_agent(state, resources_instance))
    workflow.add_node("general_conversation_node", lambda state: general_conversation_agent(state, resources_instance))
    workflow.add_node("product_recommendation_node", lambda state: product_recommendation_agent(state, resources_instance))
    workflow.add_node("query_knowledge_base_node", lambda state: query_knowledge_base_agent(state, resources_instance))
    workflow.add_node("skin_analysis_node", lambda state: skin_analysis_agent(state, resources_instance))
    workflow.add_node("product_comparison_node", lambda state: product_comparison_agent(state, resources_instance))


    # Set the entry point for the graph
    workflow.set_entry_point("classify_intent_node")

    # Define conditional edges based on intent classification
    workflow.add_conditional_edges(
        "classify_intent_node",
        # This function determines the next node based on the 'next_node' key set by the intent classifier
        lambda state: state['next_node'],
        {
            "general_conversation_agent": "general_conversation_node",
            "product_recommendation_agent": "product_recommendation_node",
            "query_knowledge_base_agent": "query_knowledge_base_node",
            "skin_analysis_agent": "skin_analysis_node",
            "product_comparison_agent": "product_comparison_node",
            # Add other intents as they are defined and mapped
        }
    )

    # Define direct edges from each agent node back to the END, as they produce a final response
    # Use the imported END constant, not the string "END"
    workflow.add_edge("general_conversation_node", END) 
    workflow.add_edge("product_recommendation_node", END)
    workflow.add_edge("query_knowledge_base_node", END)
    workflow.add_edge("skin_analysis_node", END)
    workflow.add_edge("product_comparison_node", END)

    compiled_graph = workflow.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return compiled_graph

# The graph will be compiled during the FastAPI startup event and attached to app.chatbot_graph


# --- Helper Function for Chat History Conversion ---
def convert_lc_history_to_api_history(lc_history: List[BaseMessage]) -> List[Dict[str, str]]:
    """Converts LangChain BaseMessage history to a simplified dict format for API."""
    api_history = []
    for msg in lc_history:
        if isinstance(msg, HumanMessage):
            api_history.append({"human": msg.content})
        elif isinstance(msg, AIMessage):
            api_history.append({"ai": msg.content})
    return api_history


# --- API Endpoints ---

@app.get("/", summary="Health Check")
async def root():
    """
    Root endpoint for a simple health check.
    """
    logger.info("Root endpoint hit - health check.")
    return {"message": "Skincare Chatbot API is running!"}


@app.post("/chat", response_model=ChatResponse, summary="Send a chat message to the chatbot")
async def chat_endpoint(request: ChatRequest, request_fastapi: Request):
    """
    Main endpoint for interacting with the skincare chatbot.
    Receives a user message and optional chat history, processes it through LangGraph,
    and returns a chatbot response, potentially including product recommendations or skin analysis results.
    """
    try:
        logger.info(f"Received chat request: User message='{request.user_message}'")
        logger.debug(f"Chat history from request: {request.chat_history}")
        logger.debug(f"Skin analysis result from request: {request.skin_analysis_result}")

        # Convert API chat history format to LangChain BaseMessage format for internal use
        lc_chat_history: List[BaseMessage] = []
        if request.chat_history:
            for msg_dict in request.chat_history:
                if "human" in msg_dict:
                    lc_chat_history.append(HumanMessage(content=msg_dict["human"]))
                elif "ai" in msg_dict:
                    lc_chat_history.append(AIMessage(content=msg_dict["ai"]))
        
        # Initialize the LangGraph state
        initial_state: ChatbotState = {
            "user_message": request.user_message,
            "chat_history": lc_chat_history,
            "intent": "unclassified", # Will be set by intent classifier
            "next_node": "", # Will be set by intent classifier
            "response": "", # Will be set by agents
            "product_recommendation_data": None,
            "skin_analysis_result_data": request.skin_analysis_result, # Pass existing analysis data if present
            "retrieved_docs": None,
            "comparison_products": None,
        }

        logger.debug("Invoking LangGraph with initial state.")
        # Invoke the LangGraph workflow
        final_state: ChatbotState = await app.chatbot_graph.ainvoke(initial_state)
        logger.debug(f"LangGraph execution finished. Final state: {final_state}")

        # Prepare the response for the API
        response_text = final_state.get("response", "I'm sorry, I couldn't process your request fully. Please try again.")
        
        # Update chat history with the current turn for the next request
        updated_lc_chat_history = lc_chat_history + [
            HumanMessage(content=request.user_message),
            AIMessage(content=response_text)
        ]
        api_chat_history = convert_lc_history_to_api_history(updated_lc_chat_history)

        api_product_recommendations = final_state.get("product_recommendation_data")
        if api_product_recommendations:
            logger.info(f"Returning {len(api_product_recommendations)} product recommendations.")
        
        api_skin_analysis_result = final_state.get("skin_analysis_result_data")
        if api_skin_analysis_result:
            logger.info("Returning skin analysis result.")

        logger.info("Chat request processed successfully.")
        return ChatResponse(
            response=response_text,
            chat_history=api_chat_history,
            product_recommendation=api_product_recommendations,
            skin_analysis_result=api_skin_analysis_result
        )

    except Exception as e:
        logger.error(f"Error during chat processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {e}"
        )


@app.post("/upload_image", response_model=SkinAnalysisResult, summary="Upload an image for skin analysis")
async def upload_image_endpoint(file: UploadFile = File(...)):
    """
    Receives an image file, saves it, and performs a mock skin analysis.
    Returns the skin analysis result.
    """
    logger.info(f"Received image upload request for file: {file.filename}, Content-Type: {file.content_type}")

    # 1. Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        logger.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JPEG and PNG images are allowed."
        )

    # 2. Generate a unique filename and path
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    upload_path = os.path.join(Config.UPLOAD_IMAGE_DIR, unique_filename)

    # Ensure the upload directory exists
    os.makedirs(Config.UPLOAD_IMAGE_DIR, exist_ok=True)

    # 3. Save the image
    try:
        # Save file in chunks to handle large files efficiently
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Image saved successfully to: {upload_path}")

        # 4. Perform skin analysis using the tools agent
        skin_analysis_result = tools.get_skin_type_from_image(upload_path)

        if skin_analysis_result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to perform skin analysis on the uploaded image."
            )

        logger.info(f"Skin analysis completed: Skin Type='{skin_analysis_result.skin_type}'")
        return skin_analysis_result

    except HTTPException as e:
        # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error(f"Error during image upload or analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during image processing."
        )
    finally:
        # Optional: Clean up the uploaded file after processing
        # In a production system, you might move it to a permanent storage or process it asynchronously
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
                logger.debug(f"Cleaned up uploaded image file: {upload_path}")
            except Exception as e:
                logger.error(f"Failed to remove uploaded image file '{upload_path}': {e}")
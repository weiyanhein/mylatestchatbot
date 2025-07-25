# backend/agents/general_conversation.py

from typing import Dict, Any
from backend.models import ChatbotState
from backend.agents.resources import ChatbotResources
from backend.logger_config import logger # Import the centralized logger
from langchain_core.messages import HumanMessage, AIMessage

def general_conversation_agent(state: ChatbotState, resources: ChatbotResources) -> ChatbotState:
    """
    Handles general conversation using the LLM instance dedicated to natural language output.
    This agent is a fallback for intents that don't require specialized processing.
    """
    user_message = state.get("user_message", "").strip()
    chat_history = state.get("chat_history", [])
    logger.info(f"General Conversation Agent activated for user message: '{user_message}'")

    # Access the dedicated chat LLM instance via the 'resources' object
    ollama_chat_llm = resources.ollama_chat_llm

    if ollama_chat_llm is None:
        response_text = "I'm sorry, my general conversation service is not available right now. Please try again later."
        logger.error("LLM (ollama_chat_llm) not initialized for general conversation agent.")
        state['response'] = response_text
        return state

    try:
        # Construct messages for the LLM, including recent chat history for context
        messages = [
            {"role": "system", "content": """You are a friendly and helpful skincare assistant. You can chat about general topics and provide general information, but your primary expertise is skincare. Keep your responses concise, polite, and directly address the user's query. If a query seems complex or requires specific product or knowledge base information, suggest asking a more focused question."""}
        ]

        # Add recent chat history (last 5 turns) to provide conversational context
        for msg in chat_history[-5:]: # Use the last 5 messages for context
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})

        messages.append({"role": "user", "content": user_message})

        logger.debug(f"Calling ollama_chat_llm for general conversation with messages: {messages}")
        llm_response = ollama_chat_llm.invoke(messages)
        
        # CHANGED: Access .content attribute for AIMessage objects
        response_text = llm_response.content.strip() if llm_response and hasattr(llm_response, 'content') else "I'm not sure how to respond to that. Can you please rephrase or ask a different question?"
        logger.info(f"General Conversation LLM response: {response_text[:100]}...") # Log first 100 chars

    except Exception as e:
        logger.error(f"Error in general_conversation_agent: {e}", exc_info=True)
        response_text = "I encountered an issue while trying to chat. Please try again or ask a simpler question."

    state['response'] = response_text
    return state
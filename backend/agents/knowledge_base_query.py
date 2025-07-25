# backend/agents/knowledge_base_query.py

from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from backend.models import ChatbotState
from backend.agents.resources import ChatbotResources # Import the class
from backend.logger_config import logger # Import the centralized logger

def query_knowledge_base_agent(state: ChatbotState, resources: ChatbotResources) -> ChatbotState:
    """
    Queries the knowledge base (PDFs) based on the user's message
    and uses the LLM to synthesize an answer.
    """
    user_message = state.get("user_message", "").strip()
    chat_history = state.get("chat_history", [])
    logger.info(f"Knowledge Base Query Agent activated for: '{user_message}'")

    # Access resources via the 'resources' object passed to the agent
    ollama_chat_llm = resources.ollama_chat_llm
    knowledge_base_vectorstore = resources.knowledge_base_vectorstore

    if ollama_chat_llm is None or knowledge_base_vectorstore is None:
        response_text = "I'm sorry, my knowledge base or language model is not available right now."
        logger.error("LLM (ollama_chat_llm) or knowledge base vectorstore not initialized.")
        state['response'] = response_text
        return state

    retrieved_docs = []
    try:
        # Perform similarity search on the knowledge base
        logger.debug(f"Performing similarity search for: '{user_message}'")
        retrieved_docs = knowledge_base_vectorstore.similarity_search(user_message, k=2)
        logger.info(f"Retrieved {len(retrieved_docs)} documents from knowledge base.")

        if not retrieved_docs:
            response_text = "I couldn't find specific information related to your query in my knowledge base. Can I help with something else?"
            logger.info("No relevant documents found in knowledge base for the query.")
            state['response'] = response_text
            state['retrieved_docs'] = []
            return state

        # Concatenate the content of the retrieved documents to form the context
        context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        logger.debug(f"Retrieved context from knowledge base (first 200 chars): \n{context[:200]}...")

        # Prepare messages for the LLM using the retrieved context
        messages = [
            {"role": "system", "content": f"""You are a helpful and knowledgeable skincare assistant.
            Your task is to answer the user's question using ONLY the provided context below.
            If the answer is not explicitly stated in the context, clearly state that you don't know or that the information is not available in the provided text.
            Do not make up information. Keep your answer concise and factual.

            --- Context for answering the user's question ---
            {context}
            --- End of Context ---
            """},
        ]

        # Add recent chat history for conversational context (last 3 turns)
        for msg in chat_history[-3:]:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})

        messages.append({"role": "user", "content": user_message})

        logger.debug(f"Calling ollama_chat_llm for knowledge base query.")
        llm_response = ollama_chat_llm.invoke(messages)
        # Access the string content from the AIMessage object before stripping
        response_text = llm_response.content.strip() if llm_response and hasattr(llm_response, 'content') else "I couldn't generate an answer from the available information. The context might not have been sufficient."
        logger.info(f"Knowledge Base LLM response (first 100 chars): {response_text[:100]}...")

        state['response'] = response_text
        state['retrieved_docs'] = [doc.dict() for doc in retrieved_docs] # Store retrieved docs for debugging/auditing

    except Exception as e:
        logger.error(f"Error in query_knowledge_base_agent: {e}", exc_info=True)
        response_text = "I encountered an issue while querying my knowledge base. Please try again or rephrase your question."
        state['response'] = response_text
        state['retrieved_docs'] = [] # Ensure this is cleared on error

    return state
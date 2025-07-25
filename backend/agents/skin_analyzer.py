# backend/agents/skin_analyzer.py

from typing import Dict, Any,Optional
from backend.models import ChatbotState, SkinAnalysisResult
from backend.agents.resources import ChatbotResources
from backend.logger_config import logger # Import the centralized logger
from langchain_core.messages import HumanMessage, AIMessage

def skin_analysis_agent(state: ChatbotState, resources: ChatbotResources) -> ChatbotState:
    """
    Handles skin analysis requests.
    If skin analysis data is present in the state (from an image upload), it processes and
    summarizes it. Otherwise, it prompts the user to upload an image.
    """
    user_message = state.get("user_message", "").strip()
    chat_history = state.get("chat_history", [])
    skin_analysis_result_data: Optional[SkinAnalysisResult] = state.get("skin_analysis_result_data")

    logger.info(f"Skin Analysis Agent activated for user message: '{user_message}'")
    logger.debug(f"Current skin_analysis_result_data in state: {skin_analysis_result_data}")

    ollama_chat_llm = resources.ollama_chat_llm

    if ollama_chat_llm is None:
        response_text = "I'm sorry, my language model service is not available to process analysis results."
        logger.error("LLM (ollama_chat_llm) not initialized for skin analysis agent.")
        state['response'] = response_text
        return state

    if skin_analysis_result_data:
        # If analysis result is already in the state (from a prior /upload_image call)
        logger.info("Processing existing skin analysis result.")
        skin_type = skin_analysis_result_data.skin_type
        concerns = ", ".join(skin_analysis_result_data.concerns) if skin_analysis_result_data.concerns else "no specific concerns detected"
        analysis_recommendations = skin_analysis_result_data.recommendations

        # Use LLM to summarize and give actionable advice based on the structured data
        try:
            messages = [
                {"role": "system", "content": f"""You are a helpful skincare assistant.
                You have just received the following skin analysis result:
                Skin Type: {skin_type}
                Concerns: {concerns if concerns else 'None'}
                General Recommendations: {analysis_recommendations}

                Based on this, generate a concise, encouraging, and actionable summary for the user.
                Always start by confirming the skin type.
                Suggest next steps, such as asking for product recommendations for their specific type/concerns.
                Keep it under 100 words."""},
            ]
            # No need for full chat history here, as the analysis result is the primary context
            messages.append({"role": "user", "content": f"Please summarize my skin analysis. My skin type is {skin_type} and concerns are {concerns}."})

            logger.debug("Calling ollama_chat_llm to summarize skin analysis result.")
            llm_response = ollama_chat_llm.invoke(messages)
            response_text = llm_response.strip() if llm_response else "I processed your skin analysis, but couldn't generate a summary. Could you tell me more about your skin?"
            logger.info(f"Skin Analysis LLM summary: {response_text[:100]}...")

        except Exception as e:
            logger.error(f"Error calling LLM for skin analysis summary: {e}", exc_info=True)
            response_text = (
                f"Thank you for sharing your skin analysis! Your skin type appears to be **{skin_type}** "
                f"with potential concerns like {concerns if concerns else 'none'}. "
                f"General advice: {analysis_recommendations}. "
                "I had a slight issue generating a detailed summary, but I'm ready to help with product recommendations."
            )
        # Store the analysis result back in the state, in case it's needed by subsequent agents
        state['skin_analysis_result_data'] = skin_analysis_result_data
        state['response'] = response_text
    else:
        # If no analysis result is in the state, prompt the user to upload an image
        logger.info("No skin analysis result found in state. Prompting user for image upload.")
        response_text = (
            "To analyze your skin, I need a clear photo of your face. Please use the 'Upload Image' feature on the interface. "
            "Once uploaded, I can tell you your skin type and suggest general recommendations."
        )
        state['response'] = response_text
        # Ensure any stale skin_analysis_result_data is cleared if the user's intent is to restart analysis
        state['skin_analysis_result_data'] = None

    return state
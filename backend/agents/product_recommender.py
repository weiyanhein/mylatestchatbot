# backend/agents/product_recommender.py

from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from backend.models import ChatbotState, RecommendedProductForAPI, LLMRecommendedProduct, ProductRecommendationResponse, SkinAnalysisResult
from backend.agents.resources import ChatbotResources
from backend.config import Config
from backend.logger_config import logger # Import the centralized logger
import json # Import json for robust parsing if needed, though PydanticOutputParser handles it

def product_recommendation_agent(state: ChatbotState, resources: ChatbotResources) -> ChatbotState:
    """
    Recommends skincare products based on the user's query, chat history,
    and optionally, skin analysis results. It uses RAG to fetch relevant products
    and an LLM to select and present them in a structured JSON format.
    """
    user_message = state.get("user_message", "").strip()
    chat_history = state.get("chat_history", [])
    skin_analysis_result: Optional[SkinAnalysisResult] = state.get("skin_analysis_result_data")
    logger.info(f"Product Recommendation Agent activated for user message: '{user_message}'")
    logger.debug(f"Skin analysis data for recommendation: {skin_analysis_result}")

    # Access resources
    ollama_json_llm = resources.ollama_json_llm # Use the JSON-formatted LLM
    products_vectorstore = resources.products_vectorstore
    products_collection = resources.products_collection

    if ollama_json_llm is None or products_vectorstore is None or products_collection is None:
        response_text = "I'm sorry, my product recommendation service is not fully initialized. Please try again later."
        logger.error("Required resources (ollama_json_llm, products_vectorstore, or products_collection) not initialized.")
        state['response'] = response_text
        return state

    recommended_products_for_api: List[RecommendedProductForAPI] = []
    response_general_message = "I'm sorry, I couldn't find any suitable recommendations at this moment."

    try:
        # Step 1: Retrieve relevant products from the vector store
        query_text = user_message
        if skin_analysis_result:
            # Enhance query with skin analysis details for better RAG
            query_text += f" My skin type is {skin_analysis_result.skin_type} and concerns are {', '.join(skin_analysis_result.concerns)}. I need products for these."

        logger.debug(f"Performing similarity search on products for query: '{query_text}'")
        retrieved_products_docs = products_vectorstore.similarity_search(query_text, k=Config.MAX_RECOMMENDED_PRODUCTS + 1) # Get a few extra
        logger.info(f"Retrieved {len(retrieved_products_docs)} potential products from vector store.")

        if not retrieved_products_docs:
            response_text = "I couldn't find any products matching your criteria. Can I help with something else?"
            state['response'] = response_text
            state['product_recommendation_data'] = []
            return state

        # Fetch full details for retrieved products from MongoDB
        retrieved_product_ids = [doc.metadata.get("product_id") for doc in retrieved_products_docs if doc.metadata.get("product_id")]
        if not retrieved_product_ids:
            response_text = "I found some general matches, but couldn't retrieve full product details. Please try a different query."
            state['response'] = response_text
            state['product_recommendation_data'] = []
            logger.warning("No product_ids found in retrieved documents' metadata.")
            return state

        # Remove duplicates while preserving order
        unique_product_ids = []
        seen_ids = set()
        for p_id in retrieved_product_ids:
            if p_id not in seen_ids:
                unique_product_ids.append(p_id)
                seen_ids.add(p_id)

        # Fetch products from MongoDB based on unique IDs
        db_products_cursor = products_collection.find({"product_id": {"$in": unique_product_ids}})
        db_products = list(db_products_cursor)
        logger.info(f"Fetched {len(db_products)} full product details from MongoDB.")

        if not db_products:
            response_text = "I found potential products, but couldn't retrieve their full details from the database. Please try a different query."
            state['response'] = response_text
            state['product_recommendation_data'] = []
            return state

        # Create a dictionary for quick lookup
        db_products_map = {p['product_id']: p for p in db_products}
        
        # Ensure we only pass products that have full details and limit to MAX_RECOMMENDED_PRODUCTS
        filtered_products_for_llm = []
        for p_id in unique_product_ids:
            if p_id in db_products_map:
                filtered_products_for_llm.append(db_products_map[p_id])
            if len(filtered_products_for_llm) >= Config.MAX_RECOMMENDED_PRODUCTS + 2: # Keep a few more for LLM to choose from
                break
        
        if not filtered_products_for_llm:
            response_text = "I found some relevant products but was unable to process them for recommendation. Could you try a more specific search?"
            state['response'] = response_text
            state['product_recommendation_data'] = []
            return state

        # Convert products to a string format for LLM context, focusing on key details
        products_context_str = "\n\n".join([
            f"Product ID: {p.get('product_id')}\n"
            f"Name: {p.get('name')}\n"
            f"Brand: {p.get('brand')}\n"
            f"Type: {p.get('product_type', 'N/A')}\n"
            f"Ingredients: {', '.join(p.get('ingredients', [])) if p.get('ingredients') else 'N/A'}\n"
            f"Benefits: {', '.join(p.get('benefits', [])) if p.get('benefits') else 'N/A'}\n"
            f"Usage: {p.get('usage', 'N/A')}\n"
            f"Image Filename: {p.get('image_filename', 'N/A')}"
            for p in filtered_products_for_llm
        ])
        logger.debug(f"Products context for LLM (first 200 chars): {products_context_str[:200]}...")

        # Step 2: Use LLM with PydanticOutputParser to select and format recommendations
        parser = PydanticOutputParser(pydantic_object=ProductRecommendationResponse)

        format_instructions = parser.get_format_instructions()
        logger.debug(f"Pydantic output format instructions: {format_instructions}")

        system_message_content = f"""
        You are an expert skincare product recommender. Your goal is to select the BEST {Config.MAX_RECOMMENDED_PRODUCTS} products
        from the provided 'PRODUCT INFORMATION' that most closely match the user's 'QUERY', 'CHAT HISTORY',
        and 'SKIN ANALYSIS RESULT' (if available).

        **CRITICAL INSTRUCTION: Your entire response MUST be a single, valid JSON object, and NOTHING ELSE.**
        **DO NOT include any conversational text, explanations, or markdown outside of the JSON block.**
        **The JSON MUST conform exactly to the following Pydantic schema:**
        ```json
        {format_instructions}
        ```

        --- PRODUCT SELECTION CRITERIA ---
        1.  **Relevance:** Prioritize products directly relevant to the user's explicit needs and skin analysis.
        2.  **Completeness:** Ensure the recommended products have sufficient detail to provide good reasoning.
        3.  **Diversity:** If possible, recommend different types of products (e.g., cleanser, serum, moisturizer) if the query is broad.
        4.  **Actionable Reasoning:** For each product, provide a concise but compelling `reasoning` explaining *why* it's a good fit based on its attributes (ingredients, benefits) and the user's needs. This is "smart guiding knowledge".

        --- QUERY ---
        User's current query: {user_message}

        --- CHAT HISTORY ---
        {' '.join([f"{msg.type}: {msg.content}" for msg in chat_history[-3:]]) if chat_history else 'No recent chat history.'}

        --- SKIN ANALYSIS RESULT ---
        {f"Skin Type: {skin_analysis_result.skin_type}, Concerns: {', '.join(skin_analysis_result.concerns)}, General Advice: {skin_analysis_result.recommendations}" if skin_analysis_result else 'No skin analysis result provided.'}

        --- PRODUCT INFORMATION (Choose from these products ONLY) ---
        {products_context_str}
        --- END PRODUCT INFORMATION ---

        **IMPORTANT:**
        -   Only recommend products found in the 'PRODUCT INFORMATION' section.
        -   Ensure `product_id`, `name`, `brand`, and `image_filename` for each recommended product EXACTLY match the provided `PRODUCT INFORMATION`.
        -   Generate a friendly `general_message` to introduce the recommendations.
        -   The `reasoning` for each product must be specific and helpful.
        """

        messages = [
            {"role": "system", "content": system_message_content},
            # Removed the explicit user message here as the system prompt is highly directive
            # If the LLM still struggles, you might re-add a concise "Return JSON now." here.
        ]

        logger.debug("Calling ollama_json_llm for product recommendations.")
        llm_response = ollama_json_llm.invoke(messages)
        
        # Access .content for logging
        if llm_response and hasattr(llm_response, 'content'):
            logger.debug(f"Raw LLM JSON response: {llm_response.content[:500]}...")
            # Attempt to strip any outer markdown if the LLM adds it, as a fallback
            raw_llm_json_str = llm_response.content.strip()
            if raw_llm_json_str.startswith("```json"):
                raw_llm_json_str = raw_llm_json_str[len("```json"):].strip()
            if raw_llm_json_str.endswith("```"):
                raw_llm_json_str = raw_llm_json_str[:-len("```")].strip()
        else:
            logger.debug(f"LLM response is not an AIMessage with content or is empty: {llm_response}")
            raw_llm_json_str = "" # Fallback if no content

        # Use a try-except block specifically for parsing to catch the JSON error more cleanly
        try:
            # Parse the LLM's JSON output
            # If the LLM *sometimes* includes the ```json ```, this stripping helps.
            parsed_llm_output = parser.parse(raw_llm_json_str) 
            response_general_message = parsed_llm_output.general_message
            llm_recommended_products: List[LLMRecommendedProduct] = parsed_llm_output.recommendations
        except Exception as parse_error:
            logger.error(f"Failed to parse LLM response into Pydantic model. Raw LLM content: {raw_llm_json_str[:1000]}...", exc_info=True)
            raise parse_error # Re-raise to be caught by the outer exception handler

        if not llm_recommended_products:
            response_text = response_general_message + " However, I couldn't select any specific products from the options provided. Could you try a more detailed query?"
            state['response'] = response_text
            state['product_recommendation_data'] = []
            logger.warning("LLM returned no specific product recommendations in the JSON output.")
            return state

        # Step 3: Map LLM output to API model and generate image URLs
        base_image_url_prefix = "/static/product_images/" # Matches FastAPI static files mount

        for llm_rec in llm_recommended_products:
            # Look up full details from the products fetched from MongoDB
            full_product_details = db_products_map.get(llm_rec.product_id)
            if not full_product_details:
                logger.warning(f"Product ID {llm_rec.product_id} recommended by LLM not found in fetched DB products. Skipping.")
                continue

            # Construct the image_url
            image_filename = full_product_details.get("image_filename", "default.jpg") # Fallback
            image_url = f"{base_image_url_prefix}{image_filename}"

            recommended_products_for_api.append(RecommendedProductForAPI(
                name=full_product_details.get("name"),
                brand=full_product_details.get("brand"),
                product_id=full_product_details.get("product_id"),
                image_filename=image_filename,
                image_url=image_url,
                reasoning=llm_rec.reasoning, # Use LLM's generated reasoning
                product_type=full_product_details.get("product_type"),
                ingredients=full_product_details.get("ingredients"),
                benefits=full_product_details.get("benefits"),
                usage=full_product_details.get("usage")
            ))
            if len(recommended_products_for_api) >= Config.MAX_RECOMMENDED_PRODUCTS:
                break # Ensure we don't exceed the max number of recommendations

        if not recommended_products_for_api:
            response_text = response_general_message + " Unfortunately, I couldn't finalize any recommendations based on the details."
            state['response'] = response_text
            state['product_recommendation_data'] = []
            logger.warning("No products could be converted to RecommendedProductForAPI after LLM processing.")
            return state

        response_text = response_general_message
        state['response'] = response_text
        state['product_recommendation_data'] = recommended_products_for_api
        logger.info(f"Successfully generated {len(recommended_products_for_api)} product recommendations.")
        logger.debug(f"Final product recommendations: {[p.name for p in recommended_products_for_api]}")

    except Exception as e:
        logger.error(f"Error in product_recommendation_agent: {e}", exc_info=True)
        response_text = "I encountered an issue while trying to recommend products. Please try again or provide more details."
        state['response'] = response_text
        state['product_recommendation_data'] = [] # Clear any partial data on error

    return state
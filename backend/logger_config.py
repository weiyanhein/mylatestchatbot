# backend/logger_config.py
import logging
import os
from backend.config import Config

def setup_logging():
    """
    Sets up the logging configuration for the application.
    Reads the log level from Config.LOG_LEVEL.
    """
    log_level_str = Config.LOG_LEVEL.upper()
    numeric_log_level = getattr(logging, log_level_str, logging.INFO)

    # Configure basic logging for the application
    logging.basicConfig(
        level=numeric_log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler() # Outputs logs to console
        ]
    )

    # Suppress verbose logs from external libraries
    logging.getLogger("uvicorn").propagate = False # Prevent uvicorn from double logging if it's already handled
    logging.getLogger("uvicorn.access").propagate = False
    logging.getLogger("langchain_core").setLevel(logging.WARNING) # Suppress verbose langchain_core logs
    logging.getLogger("httpx").setLevel(logging.WARNING) # Suppress httpx library logs

    # Return a specific logger instance for your application
    app_logger = logging.getLogger("skincare_chatbot_app")
    app_logger.setLevel(numeric_log_level) # Ensure the app logger uses the configured level
    return app_logger

# Initialize the logger immediately when this module is imported
logger = setup_logging()
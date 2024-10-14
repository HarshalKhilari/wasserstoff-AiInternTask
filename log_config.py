import logging


# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),   # Log to a file
        logging.StreamHandler()           # Log to console
    ]
)

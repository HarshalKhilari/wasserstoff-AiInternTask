import os
import logging
import log_config
from constants import DIRECTORY
from mongoDB_setup import metadata_collection
from file_and_db_ops import import_files_concurrently
from summary_and_keywords import summary_keyword_extract_concurrently


def check_if_new_files():
    """Detects and imports new files if they exist."""
    try:

        # Get list of files in the directory
        files_in_directory = os.listdir(DIRECTORY)
        logging.info(f"Found {len(files_in_directory)} files in the directory.")

        # Get list of filenames in MongoDB
        db_files = metadata_collection.find({}, {"file_name": 1})  # Adjust the field name as per your database
        db_file_names = {file['file_name'] for file in db_files}  # Use a set for faster lookups

        # Identify new files that are not in the database
        new_files = [file for file in files_in_directory if file not in db_file_names]

        # Check if new files exist
        if new_files:
            logging.info(f"Found the following new files: {new_files}")

            # Import files concurrently
            import_files_concurrently(new_files)

            # Call summary and keyword extraction function after importing files
            summary_keyword_extract_concurrently(new_files)
        else:
            logging.info("No new files found.")
    except Exception as err:
        logging.error(f"An error occurred while checking for new files: {err}")
from constants import DIRECTORY
from mongoDB_setup import metadata_collection
import os
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import log_config


def import_files_concurrently(file_name_list):
    """Imports a list of files concurrently."""
    cpus = os.cpu_count()   # Getting total virtual CPUs
    ncpu = max(1, cpus - 2) if cpus > 2 else 1  # Leaving 1 CPU for other processes
    with ThreadPoolExecutor(max_workers=ncpu) as executor:
        futures = {executor.submit(import_single_file, file_name): file_name for file_name in file_name_list}
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                future.result()  # This will raise an exception if import failed
            except Exception as err:
                logging.error(f"Error importing {file_name}: {str(err)}")


def import_single_file(file_name):
    """Imports a single file and its metadata into the database."""
    try:
        file_path = os.path.join(DIRECTORY, file_name)
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            if pdf_reader.is_encrypted:
                pdf_reader.decrypt('')  # Try decrypting if needed

            metadata = pdf_reader.metadata
            error = 'n/a'

            # Store metadata in DB
            store_metadata_in_db(file_name, metadata, error)
            logging.info(f"Successfully imported file: {file_name}")
    except Exception as err:
        error = str(err)
        logging.error(f"Error importing file {file_name}: {error}")
        # Store metadata even if there's an issue
        store_metadata_in_db(file_name, None, error)


def store_metadata_in_db(file_name, metadata, error):
    document = {
        "file_name": file_name,
        "metadata": metadata,
        "errors": error
    }
    try:
        metadata_collection.insert_one(document)
        logging.info(f"Metadata stored in the database for file: {file_name}")
    except Exception as err:
        logging.error(f"Error storing metadata for {file_name}: {err}")

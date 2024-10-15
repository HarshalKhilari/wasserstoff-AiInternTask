import logging
import log_config
from talk2db import check_if_new_files
from reporting import generate_report
from mongoDB_setup import *


def launch():
    try:
        logging.info("Starting the file processing pipeline...")
        check_if_new_files()
        logging.info("File processing completed.")
        generate_report()
    except Exception as e:
        logging.error(f"An error occurred in the launch process: {e}")


if __name__ == '__main__':
    launch()
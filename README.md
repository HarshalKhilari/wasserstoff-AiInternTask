# wasserstoff-AiInternTask
AI Internship Task Submission for Qualification


# PDF Summarization and Keyword Extraction Pipeline

## Overview

This project implements a dynamic pipeline for processing multiple PDF documents, generating summaries and keywords, and storing them in a MongoDB database. The solution prioritizes concurrency and speed, leveraging Python's multiprocessing capabilities to efficiently handle large volumes of documents.

## Problem Statement

With the increasing amount of information available in PDF format, it is essential to develop tools that can extract meaningful content quickly. This project aims to automate the summarization and keyword extraction from PDF documents, allowing users to quickly grasp the main ideas without having to read through entire files.

## Clone the Repository

   ```bash
   git clone https://github.com/HarshalKhilari/wasserstoff-AiInternTask.git
   cd wasserstoff-AiInternTask
   ```

## System Requirements

- Minimum 4 GB RAM
- CPU from within 10 years
- GPU (for faster processing speed; not mandatory)

## Dependencies

Install required libraries using the following code in cmd:
```
pip install -r requirements.txt
```

## Run the Pipeline:
Place your PDF files in the specified input directory 'pdf_store'.

Execute the main script:
```
python launch.py
```

## Explanation of the Solution

The project consists of several modules that work together to achieve the desired functionality:
	- launch.py: Launching point for the app
    - summary_and_keywords.py: Contains functions to clean text, split text into manageable chunks, summarize those chunks, and extract keywords.
    - mongoDB_setup.py: Handles the connection and setup for the MongoDB database.
    - file_and_db_ops.py: Manages file operations (reading PDF files) and interactions with the database (inserting summaries and keywords).
    - talk2db.py: Interfaces with the MongoDB database for data retrieval and storage.
    - reporting.py: Generates performance reports for the processing pipeline.

Key Features

    - Concurrency: Utilizes Python's multiprocessing capabilities to process multiple PDF documents simultaneously, significantly improving performance.
    - Summary Generation: Summarizes long PDF documents into concise summaries, providing users with a quick overview.
    - Keyword Extraction: Identifies and extracts key terms from the documents to facilitate easier navigation and searchability.
	
Additional details:
	- Implemented thread level concurrency for importing pdf file metadata into the database
	- When the app is launched, it checks for files in the folder that are not in the database and processes them
	- The method for ingesting different fie lengths is the same
	- pdf metadata, summary, keywords, time and performance metrics are updated to the database
	- Transformers pipeline is used to summarize the text after cleaning to remove tables and other structured data. "sshleifer/distilbart-cnn-12-6" is used for small and medium length files, and "facebook/bart-base" for large files. summary sizes are split into 3 catagories based on the token size for the cleaned text
	- KeyBERT model was used for keyword extraction.
	- Document processing pipeline included extracting text from pdf, cleaning text, summarizing it, keyword extraction, storing relevant information into the database. Concurrency was implemented for pipeline on process level
	- Error handling and Logging implemented throughout the project
	- Reports are generated to show performance metrics stored in the database based on time taken to perform certain task that use parallelization, as well as system resource monitoring
	
	
Limitations faced:
	- Document processing pipeline could not be highly parallel as the memory requirements exceeded the available memory and lead to memory overflow and other memory-related errors leading to app crashing. Attempt to calculate available memory and assign processes accordingly failed due to a bug which couldn't be resolved.
	- The light-weight models chosen still took a long time and extensive memory (nearly 1 GB per process) for summarization task
	- The keyword extraction model has given limited quality of results
	- ChapGPT and library documentation was used for code generation, correction, and refinement.
	- Testing and Docker Setup were not implemented due to time constraints
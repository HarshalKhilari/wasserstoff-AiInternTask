from constants import DIRECTORY
import re
import os
import time
import psutil
import logging
import log_config
import fitz
from transformers import pipeline
from concurrent.futures import ProcessPoolExecutor, as_completed
from mongoDB_setup import metadata_collection
import gc
from transformers import AutoTokenizer
from keybert import KeyBERT
from itertools import permutations


# Main function for summary and keyword extraction
def summary_keyword_extract_concurrently(file_name_list):
    """Extracts text, summarizes, and extracts keywords for multiple PDFs concurrently."""

    # Get total non-hardware reserved memory in bytes
    non_reserved_memory = psutil.virtual_memory().total
    # Calculate the maximum number of workers based on 60% of non-reserved memory
    max_memory_for_tasks = 0.6 * non_reserved_memory  # 60% of non-hardware reserved memory
    max_workers = max(1, int(max_memory_for_tasks / (1024 ** 3)))  # Convert bytes to GB and ensure at least 1 worker

    logging.info(f"Total Memory: {non_reserved_memory / (1024 ** 3):.2f} GB")
    logging.info(f"Max Workers set to: {max_workers}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_pdf, file_name): file_name for file_name in file_name_list}
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                future.result()  # Raises an exception if the task failed
            except Exception as err:
                logging.error(f"Error processing {file_name}: {err}")


def process_single_pdf(file_name):
    """Process a single PDF file - extract text, summarize, and extract keywords."""
    try:
        # Step 1: Extract Text
        logging.info(f"Extracting text for file {file_name}")
        pdf_extraction_start_time = time.time()
        raw_text = extract_text_from_pdf(file_name)
        pdf_extraction_end_time = time.time()
        logging.info(
            f"Extracted text for file {file_name} in {pdf_extraction_end_time - pdf_extraction_start_time} seconds"
        )

        # Step 2: Clean Text
        cleaned_text = clean_text(raw_text, file_name)

        # Step 3: Summarize Text and add to database
        summary_start_time = time.time()

        # Track CPU and Memory before summarization
        process = psutil.Process(os.getpid())
        cpu_before = process.cpu_percent(interval=None) # divide by psutil.cpu_count() to emulate windows task manager
        memory_before = process.memory_info().rss  # in bytes

        summary, model_name, total_tokens = summarize_text(cleaned_text, file_name)

        # Track CPU and Memory after summarization
        cpu_after = process.cpu_percent(interval=None) # divide by psutil.cpu_count() to emulate windows task manager
        memory_after = process.memory_info().rss  # in bytes
        summary_end_time = time.time()
        logging.info(
            f"Summarization for file {file_name} took {summary_end_time - summary_start_time} seconds"
        )

        metadata_collection.update_one(
            {"file_name": file_name},
            {"$set": {
                "pdf_extraction_start_time": pdf_extraction_start_time,
                "pdf_extraction_end_time": pdf_extraction_end_time,
                "total_pdf_extraction_time": pdf_extraction_end_time - pdf_extraction_start_time,
                "summary": summary,
                "model": model_name,
                "token_count": total_tokens,
                "summary_start_time": summary_start_time,
                "summary_end_time": summary_end_time,
                "total_summary_time": summary_end_time - summary_start_time,
                "cpu_after": cpu_after,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_used": memory_after - memory_before  # Memory used during summarization
            }}
        )
        logging.info(f"Processed {file_name}: Summary stored in DB.")

        # Step 4: Extract Keywords and add to database
        keyword_start_time = time.time()
        keywords = extract_keywords(cleaned_text, file_name)
        keyword_end_time = time.time()
        logging.info(
            f"Keyword extraction for file {file_name} took {keyword_end_time - keyword_start_time} seconds"
        )
        metadata_collection.update_one(
            {"file_name": file_name},
            {"$set": {
                "keywords": keywords,
                "keyword_start_time": keyword_start_time,
                "keyword_end_time": keyword_end_time,
                "total_keyword_time": keyword_end_time - keyword_start_time
            }}
        )
        logging.info(f"Processed {file_name}: Keywords stored in DB.")

    except Exception as err:
        logging.error(f"Error processing file {file_name}: {err}")


# Extract text from PDF using PyMuPDF
def extract_text_from_pdf(file_name):
    try:
        pdf_doc = fitz.open(DIRECTORY + '\\' + file_name)
        full_text = ""
        for page in pdf_doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    block_text = ""
                    num_lines = len(block["lines"])
                    block_height = block["bbox"][3] - block["bbox"][1]
                    avg_line_height = block_height / num_lines if num_lines > 0 else 0
                    block_width = block["bbox"][2] - block["bbox"][0]
                    if block_width > 200 and avg_line_height > 10:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"] + " "
                    if block_text.strip():
                        full_text += block_text
        return full_text
    except Exception as err:
        logging.error(f"Error extracting text from {file_name}: {err}")
        return ""


# 2. Clean the extracted text by removing headers, footers, and page numbers
def clean_text(text, file_name):
    """Cleans the extracted text by removing headers, footers, and page numbers."""
    try:
        logging.info(f"Starting text cleaning process for file {file_name}.")
        # Remove page numbers like '1 of 10'
        text = re.sub(r"\d+\s+of\s+\d+", "", text)

        # Split the text into lines
        lines = text.split("\n")

        # Set to track unique lines
        seen_lines = set()
        cleaned_lines = []

        for line in lines:
            stripped_line = line.strip()
            # Skip empty lines, digits (page numbers), or repeated lines
            if stripped_line and not stripped_line.isdigit() and stripped_line not in seen_lines:
                cleaned_lines.append(stripped_line)
                seen_lines.add(stripped_line)  # Mark this line as seen

        # Join the cleaned lines into a single string
        cleaned_text = " ".join(cleaned_lines)
        logging.info(f"Text cleaning process completed successfully for file {file_name}.")
        return cleaned_text

    except Exception as err:
        logging.error(f"Error during text cleaning for file {file_name}: {str(err)}")
        raise


def summarize_text(txt, file_name):
    """Summarizes a given text based on the number of tokens."""
    try:
        logging.info(f"Starting summarization process for file {file_name}.")

        # Step 1: Count the number of tokens in the text
        # We use the distilbart tokenizer to get an initial estimate of the token count
        distilbart_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        token_count = count_tokens(txt, distilbart_tokenizer)  # Assuming distilbart for initial token counting

        model_name = "sshleifer/distilbart-cnn-12-6"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_length = 96
        min_length = 4

        # Step 3: Initialize the summarizer with the selected model
        summarizer = pipeline("summarization", model=model_name, tokenizer=tokenizer)

        # Split the text into manageable chunks for summarization
        max_chunk_length = 512
        text_chunks = split_text_by_length(txt, max_chunk_length, file_name)
        logging.info(f"Total number of chunks in file {file_name}: {len(text_chunks)}")

        # Summarize each chunk sequentially
        summaries = []
        for idx, chunk in enumerate(text_chunks):
            summary = summarize_chunk(chunk, summarizer, idx, file_name, max_length, min_length)
            summaries.append(summary)

        final_summary = " ".join(summaries)
        logging.info(f"Summarization process completed successfully for file {file_name}.")

        # Clear memory
        del summarizer
        gc.collect()  # Force garbage collection to release memory

        return final_summary, model_name, token_count

    except Exception as err:
        logging.error(f"Error during summarization for file {file_name}: {str(err)}")
        raise


# Function to count tokens
def count_tokens(text, tokenizer):
    """Counts the number of tokens in the given text using the specified tokenizer."""
    try:
        # Check if text is None or empty
        if text is None or text.strip() == "":
            logging.warning("Received empty text for token counting.")
            return 0  # Return 0 for empty text

        # Count tokens
        token_cnt = len(tokenizer.encode(text, truncation=False))
        return token_cnt
    except Exception as e:
        logging.error(f"Error counting tokens: {str(e)}")
        return 0  # Return 0 in case of an error


def split_text_by_length(txt, max_length, file_name):
    """Helper function to split text into chunks of roughly `max_length` words."""
    try:
        logging.info(f"Splitting text into chunks of {max_length} words for file {file_name}.")
        words = txt.split()
        chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
        logging.info(f"Text split into {len(chunks)} chunks for file {file_name}.")
        return chunks

    except Exception as err:
        logging.error(f"Error while splitting text for file {file_name}: {str(err)}")
        raise


def summarize_chunk(chunk, summarizer, chunk_index, file_name, max_length, min_length):
    """Helper function to summarize a single chunk."""
    try:
        logging.info(f"Processing chunk {chunk_index + 1} for file {file_name}.")
        st = time.time()

        # Summarize the chunk with the chosen model's max/min length
        summary_part = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)

        logging.info(f"Chunk {chunk_index + 1} processed in {time.time() - st:.2f} seconds for file {file_name}.")
        return summary_part[0]['summary_text']

    except Exception as e:
        logging.error(f"Error processing chunk {chunk_index + 1} for file {file_name}: {str(e)}")
        return ""


def extract_keywords(text, file_name, num_keywords=10):
    """Extracts keywords using the KeyBERT model and avoids repeated permutations of word combinations."""
    try:
        logging.info(f"Extracting {num_keywords} keywords from text for file {file_name} using KeyBERT.")

        kw_model = KeyBERT()
        # Extract up to trigrams (keyphrase_ngram_range=(1, 3))
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english',
                                             top_n=num_keywords * 3)  # Get more to filter later

        # Remove permutations (e.g., 'word1 word2' and 'word2 word1') and limit word occurrences
        unique_keywords = []
        seen_phrases = set()
        word_count = {}

        for kw in keywords:
            phrase = kw[0]
            words = phrase.split()
            sorted_words = tuple(sorted(words))  # Sort words to avoid permutations

            # Check if the phrase is unique and count word occurrences
            if sorted_words not in seen_phrases:
                seen_phrases.add(sorted_words)

                # Count the occurrences of each word in the phrase
                for word in words:
                    word_count[word] = word_count.get(word, 0) + 1

                # Check if adding this phrase would exceed the limit for any word
                if all(word_count[word] <= 3 for word in words):
                    unique_keywords.append(phrase)

                # Stop if we've collected enough keywords
                if len(unique_keywords) >= num_keywords:
                    break

        logging.info(f"Extracted unique keywords for file {file_name}: {unique_keywords}")
        return unique_keywords

    except Exception as err:
        logging.error(f"Error extracting keywords for file {file_name} using KeyBERT: {str(err)}")
        raise



import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mongoDB_setup import *
import logging
import log_config
import numpy as np
import os


# Truncate file names: first 5 characters, '...', and last 5 characters (if length > 10)
def shorten_name(name):
    if len(name) > 10:
        return f"{name[:5]}...{name[-5:]}"
    else:
        return name


def generate_report():
    # Load data from MongoDB into a DataFrame
    data = list(metadata_collection.find({}, {"_id": 0}))  # Exclude MongoDB's _id field
    df = pd.DataFrame(data)
    # Remove the .pdf extension and truncate file names to 10 characters, adding '...' if needed
    df['file_name'] = df['file_name'].str.replace('.pdf', '', regex=False)
    # Shorten file names
    df['short_name'] = df['file_name'].apply(shorten_name)
    # Create a new PDF document
    with PdfPages('performance_report.pdf') as pdf:


        # Plot of PDF Extraction Time
        # Set the figure size
        plt.figure(figsize=(10, 6))
        # Create the bar plot using the truncated file names
        plt.bar(df['short_name'], df['total_pdf_extraction_time'], color='blue')
        # Set plot titles and labels
        plt.title('PDF Extraction Time')
        plt.xlabel('File Name')
        plt.ylabel('Time (seconds)')
        # Rotate x-axis labels to avoid overlap, reduce font size
        plt.xticks(rotation=45, ha='right', fontsize=10)
        # Adjust layout to fit labels
        plt.tight_layout()
        # Save the first plot to the PDF
        pdf.savefig()
        plt.close()

        # Plot of Keyword Extraction Time
        plt.figure(figsize=(10, 6))  # Set the figure size
        plt.bar(df['short_name'], df['total_keyword_time'], color='orange')
        plt.title('Keyword Extraction Time')
        plt.xlabel('File Name')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()

        # Save the second plot to the PDF
        pdf.savefig()
        plt.close()

        # Plot of Summary Time
        plt.figure(figsize=(10, 6))  # Set the figure size
        plt.bar(df['short_name'], df['total_summary_time'], color='green')
        plt.title('Summary Time')
        plt.xlabel('File Name')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()

        # Save the third plot to the PDF
        pdf.savefig()
        plt.close()

        # Plot of CPU Usage for Summarization
        # Divide the 'cpu_after' column by no. of logical CPU cores
        df['cpu_usage_avg'] = df['cpu_after'] / os.cpu_count()
        plt.figure(figsize=(10, 6))
        # Create bar plot for 'cpu_usage_avg'
        plt.bar(df['short_name'], df['cpu_usage_avg'], color='green')
        # Set plot titles and labels
        plt.title('CPU Usage during Summarization')
        plt.xlabel('File Name')
        plt.ylabel('Average CPU Usage')
        # Rotate x-axis labels to avoid overlap, reduce font size
        plt.xticks(rotation=45, ha='right', fontsize=10)
        # Adjust layout to fit labels
        plt.tight_layout()

        # Save the second plot to the PDF
        pdf.savefig()
        plt.close()

        # Plot of Memory Usage During Summarization
        # Convert memory used (assuming it's in bytes) to gigabytes (GB)
        df['memory_used_gb'] = df['memory_used'] / (1024 ** 3)  # Dividing by 1024^3 to convert to GB
        # Set the figure size
        plt.figure(figsize=(10, 6))
        # Create bar plot for memory used in GB
        plt.bar(df['short_name'], df['memory_used_gb'], color='purple')
        # Set plot titles and labels
        plt.title('Memory Used during Summarization')
        plt.xlabel('File Name')
        plt.ylabel('Memory Used (GB)')
        # Rotate x-axis labels to avoid overlap, reduce font size
        plt.xticks(rotation=45, ha='right', fontsize=10)
        # Adjust layout to fit labels
        plt.tight_layout()

        # Save the third plot to the PDF
        pdf.savefig()
        plt.close()


        # Plot of Concurrency Visualization (Start and End Times)
        # Subtract the minimum y value from both 'pdf_extraction_start_time' and 'summary_end_time'
        min_y_value = min(df['pdf_extraction_start_time'].min(), df['summary_end_time'].min())
        df['adjusted_extraction_start_time'] = df['pdf_extraction_start_time'] - min_y_value
        df['adjusted_keyword_end_time'] = df['keyword_end_time'] - min_y_value
        # Convert adjusted times to minutes
        df['adjusted_extraction_start_time_min'] = df['adjusted_extraction_start_time'] / 60
        df['adjusted_keyword_end_time_min'] = df['adjusted_keyword_end_time'] / 60
        # Set the figure size
        plt.figure(figsize=(10, 6))
        # Plot the adjusted scatter plots in minutes
        plt.scatter(df['short_name'], df['adjusted_extraction_start_time_min'], label='Extraction Start Time (min)')
        plt.scatter(df['short_name'], df['adjusted_keyword_end_time_min'], label='Keyword End Time (min)', color='red')
        # Set plot titles and labels
        plt.title('PDF Processing Start and End Times')
        plt.xlabel('File Name')
        plt.ylabel('Time (Minutes)')
        # Set y-ticks to show 5-minute intervals
        plt.yticks(
            np.arange(0, df[['adjusted_extraction_start_time_min', 'adjusted_keyword_end_time_min']].max().max() + 5,
                      5))
        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        # Add a legend
        plt.legend()
        # Adjust layout to fit labels
        plt.tight_layout()

        # Save the fourth plot to the PDF
        pdf.savefig()
        plt.close()

    # The PDF file will be created as 'performance_report.pdf' in the working directory.
    logging.info("Report saved as 'performance_report.pdf'")
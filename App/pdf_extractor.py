import fitz  # PyMuPDF
import re
import os

def extract_text_excluding_toc(pdf_path):
    """Extract text from a PDF while excluding the Table of Contents (ToC)."""
    # Initialize variables
    cleaned_text = ""
    start_page = None
    end_page = None
    
    # Define regex patterns
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'\(?\b\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b'  # Simple phone pattern
    header_footer_pattern = r'^[A-Z0-9 .,-]+$'  # Basic pattern for headers/footers
    
    # Open the PDF document
    document = fitz.open(pdf_path)

    # Find the start and end pages of the TOC
    for page_num in range(len(document)):
        page = document[page_num]
        page_text = page.get_text("text")

        # Check for 'Inhaltsübersicht' to determine TOC section
        if 'Inhaltsübersicht' in page_text:
            start_page = page_num
            # Assuming the TOC ends on the next page
            end_page = start_page + 1
            break

    # If TOC was found, extract text outside of the TOC section
    for page_num in range(len(document)):
        page = document[page_num]
        page_text = page.get_text("text")

        # Only add text if it is outside the TOC section
        if (start_page is None or page_num < start_page) or (end_page is None or page_num > end_page):
            lines = page_text.split('\n')
            for line in lines:
                # Remove lines matching email, phone, and header/footer patterns
                if (re.search(email_pattern, line) or 
                    re.search(phone_pattern, line) or 
                    re.match(header_footer_pattern, line) or
                    line.strip() == ''):  # Skip empty lines
                    continue  # Skip these lines
                cleaned_text += line + "\n"  # Keep the clean lines

    # Close the document
    document.close()
    
    return cleaned_text.strip()  # Return cleaned text without trailing newlines

def extract_main_content(folder_path):
    """Extract and combine cleaned text from all PDF files in the specified folder."""
    combined_cleaned_text = ""
    
    # Iterate over all PDF files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {pdf_path}")  # Print the current PDF being processed
            cleaned_text = extract_text_excluding_toc(pdf_path)
            combined_cleaned_text += cleaned_text + "\n\n"  # Add cleaned text to the combined result

    return combined_cleaned_text.strip()  # Return combined text without trailing newlines

if __name__ == "__main__":
    pdfs_dir = os.path.join(os.path.dirname(__file__), "pdfs")  # Adjust to your PDF path
    combined_content = extract_main_content(pdfs_dir)  # Call extract_main_content with the correct folder path

    print(f"Combined extracted content:\n{combined_content[:200]}...")  # Print first 200 characters of combined content


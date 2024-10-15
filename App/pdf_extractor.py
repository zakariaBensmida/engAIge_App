import fitz  # PyMuPDF
import re
import os

def is_toc_line(line: str) -> bool:
    toc_patterns = [
        r'^\d+\s*-\s*\d+$',
        r'^[IVXLCDM]+\.$',
        r'^\d+\.$',
        r'^[a-z]+\)$',
        r'^\(\d+\)',
        r'^[A-ZÄÖÜ]\.\s',
        r'^Seite[:\s]*\d+$',
        r'^\w+\s+\d+\s*-\s*\d+$',
        r'^S\.\s*\d+$',
        r'^\d+/\d+$',
        r'^Seite\s+\d+$',
        r'^\d+\s+[A-Za-zÄÖÜäöüß]+',
    ]

    for pattern in toc_patterns:
        if re.match(pattern, line.strip()):
            return True
    return False

def is_main_title_line(line: str) -> bool:
    if len(line) < 10:
        return False
    if re.match(r'^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\s\-&.,()§]+(:)?$', line.strip()):
        if not re.search(r'\d', line):
            return True
    return False

def extract_main_content(pdf_path: str) -> str:
    """
    Extracts main content from a PDF, excluding the Table of Contents (ToC).
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted main content.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        full_text += page.get_text("text") + "\n"  # Get all text from the page
    doc.close()
    
    # Split at 'Inhaltsübersicht' (Assuming the PDF is in German)
    sections = full_text.split('Inhaltsübersicht')
    if len(sections) < 2:
        print("No 'Inhaltsübersicht' found in the document.")
        return full_text  # Return full text if ToC not found
    
    post_toc_text = sections[1]
    lines = post_toc_text.split('\n')
    
    extracted_data = []
    current_title = None
    current_content = []
    capturing = False
    
    toc_end_pattern = re.compile(r'^Seite[:\s]*\d+$', re.IGNORECASE)
    
    for line in lines:
        stripped_line = line.strip()
        
        if not capturing:
            if toc_end_pattern.match(stripped_line):
                capturing = True
            continue  # Skip lines until end of ToC
        
        # Skip residual ToC lines
        if is_toc_line(stripped_line):
            continue
        
        # Identify main titles
        if is_main_title_line(stripped_line):
            if current_title and current_content:
                extracted_data.append({
                    'title': current_title,
                    'content': ' '.join(current_content)
                })
            current_title = stripped_line
            current_content = []
        else:
            if current_title:
                current_content.append(stripped_line)
    
    # Append the last section
    if current_title and current_content:
        extracted_data.append({
            'title': current_title,
            'content': ' '.join(current_content)
        })
    
    # Combine all content
    combined_content = "\n".join([section['content'] for section in extracted_data])
    
    # Debug: Print combined content for verification
    print("Combined extracted content:\n", combined_content)

    return combined_content


if __name__ == "__main__":
    pdfs_dir = os.path.join(os.path.dirname(__file__), "pdfs")  # Adjust to your PDF path
    pdf_files = [f for f in os.listdir(pdfs_dir) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdfs_dir, pdf_file)
        print(f"Processing: {pdf_file}")
        extracted_content = extract_main_content(pdf_path)
        print(f"Extracted content from {pdf_file}:\n{extracted_content[:200]}...")  # Print first 200 characters


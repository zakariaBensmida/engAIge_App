# app/pdf_extractor.py
import fitz  # PyMuPDF
import re

def is_toc_line(line: str) -> bool:
    """
    Determines if a line is part of the Table of Contents based on common patterns.
    """
    toc_patterns = [
        r'^\d+\s*-\s*\d+$',          # e.g., "1 - 283", "73 - 189"
        r'^[IVXLCDM]+\.$',           # e.g., "I.", "II.", etc.
        r'^\d+\.$',                  # e.g., "1.", "2.", etc.
        r'^[a-z]+\)$',               # e.g., "a)", "b)", etc.
        r'^\(\d+\)',                 # e.g., "(1)", "(2)", etc.
        r'^[A-ZÄÖÜ]\.\s',            # e.g., "A. Private Altersvorsorge"
        r'^Seite[:\s]*\d+$',         # e.g., "Seite 3", "Seite: 4"
        r'^\w+\s+\d+\s*-\s*\d+$',    # e.g., "Private Altersvorsorge 1 - 283"
        r'^S\.\s*\d+$',              # e.g., "S. 3"
        r'^\d+/\d+$',                # e.g., "1/3"
        r'^Seite\s+\d+$',            # e.g., "Seite 3" (alternative pattern)
        r'^\d+\s+[A-Za-zÄÖÜäöüß]+',  # e.g., "1 Förderung", "2 Altersvorsorge"
    ]

    for pattern in toc_patterns:
        if re.match(pattern, line.strip()):
            return True
    return False

def is_main_title_line(line: str) -> bool:
    """
    Determines if a line is a main content title based on capitalization and structure.
    """
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
        full_text += page.get_text() + "\n"
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
    return combined_content


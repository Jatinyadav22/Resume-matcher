import re

def clean_resume_text(text: str) -> str:
    """
    Clean and preprocess extracted resume or job description text.
    - Normalize experience ranges (e.g., "2-5 years" → "2 to 5 years")
    - Remove emails, phone numbers, URLs
    - Remove noisy headers or common junk
    - Normalize whitespace, punctuation, and casing
    """

    # Convert to lowercase
    text = text.lower()

    # Replace common experience patterns like 2-5 years or 3–7 years
    text = re.sub(r'(\d+)[\-\–](\d+)\s*years', r'\1 to \2 years', text)

    # Remove email addresses
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', text)

    # Remove phone numbers (common patterns)
    text = re.sub(r'\b(\+?\d{1,3})?[\s\-\.]?\(?\d{2,4}\)?[\s\-\.]?\d{3,5}[\s\-\.]?\d{3,5}\b', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove common junk/resume sections
    text = re.sub(r'\b(resume|references|about me|curriculum vitae|cv)\b', '', text)

    # Replace newlines, tabs, etc.
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

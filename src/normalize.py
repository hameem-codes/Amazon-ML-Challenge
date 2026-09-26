import re
import unicodedata
import pandas as pd
import math

def normalize_text(text):
    if pd.isna(text) or text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    
    text = str(text)
    
    # Convert Unicode text into a consistent representation
    text = unicodedata.normalize('NFKC', text)
    
    # Case normalization (Unicode-safe)
    text = text.casefold()
    
    # Replace punctuation and symbols with spaces.
    # We must preserve Letters (L), Marks (M - crucial for Hindi/Tamil vowels), Numbers (N).
    chars = []
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith('P') or cat.startswith('S') or cat.startswith('C'):
            chars.append(' ')
        else:
            chars.append(char)
    text = ''.join(chars)
    
    # Collapse repeated whitespace and trim leading/trailing
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_business_name(value):
    return normalize_text(value)

def normalize_business_address(value):
    return normalize_text(value)

def normalize_country(value):
    return normalize_text(value)

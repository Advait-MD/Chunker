import os
import sys
import hashlib
from typing import List, Dict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def compute_document_hash(file_path: str) -> str:
    """Compute a deterministic SHA-256 hash of the file content (first 16 hex characters)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def read_pdf(file_path: str) -> str:
    """Read PDF file and extract text using PyPDF2."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        import PyPDF2
        print(f"Reading PDF: {file_path}")
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            print(f"PDF has {len(reader.pages)} pages")
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                    if (page_num + 1) % 50 == 0:
                        print(f"  Processed {page_num + 1}/{len(reader.pages)} pages")
        
        print(f"Extracted {len(text):,} characters")
        return text
        
    except ImportError:
        print("PyPDF2 not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'PyPDF2'])
        import PyPDF2
        return read_pdf(file_path)


def create_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Create overlapping chunks from text (paragraph-aware with sliding window fallback)."""
    chunks = []
    step = chunk_size - overlap
    
    paragraphs = text.split('\n\n')
    
    if len(paragraphs) > 1 and len(text) > 10000:
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunk_id = f"chunk_{chunk_index:06d}"
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_index,
                    'level': 0
                })
                chunk_index += 1
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        if current_chunk:
            chunk_id = f"chunk_{chunk_index:06d}"
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': chunk_id,
                'chunk_index': chunk_index,
                'level': 0
            })
    else:
        for i in range(0, len(text), step):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                chunk_id = f"chunk_{i:06d}"
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'position': i,
                    'chunk_index': len(chunks),
                    'level': 0
                })
    
    print(f"Created {len(chunks)} base chunks")
    return chunks

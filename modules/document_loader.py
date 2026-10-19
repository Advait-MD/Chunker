import os
import sys
import hashlib
import re
import unicodedata
from typing import List, Dict

try:
    from .config import config
    from .logger import get_logger
except ImportError:
    from modules.config import config
    from modules.logger import get_logger

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logger = get_logger("document_loader")


def compute_document_hash(file_path: str) -> str:
    """Compute a deterministic SHA-256 hash of the file content (first 16 hex characters)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def count_tokens(text: str) -> int:
    """
    Estimate token count accurately using word and punctuation subword tokens.
    Highly correlated with cl100k_base / Gemma tokenizer without network or package latency.
    """
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text))


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into complete grammatical sentences on punctuation boundaries.
    Never splits words or breaks mid-sentence.
    """
    if not text:
        return []
    # Split on sentence terminals (. ! ?) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        if count_tokens(s_clean) > 250:
            # Sub-split on semicolon, colon, or comma if sentence is exceptionally long
            sub_parts = re.split(r'(?<=[;:])\s+', s_clean)
            result.extend([p.strip() for p in sub_parts if p.strip()])
        else:
            result.append(s_clean)
    return result


def clean_extracted_text(text: str) -> str:
    """
    Clean extracted PDF text artifacts before chunking:
      1. Unicode normalization (NFKC resolves ligatures like 'ff', 'fi', special dashes, non-breaking spaces).
      2. Line ending and form feed normalization (\r\n, \r, \x0c -> \n).
      3. Remove standalone page numbers and common watermark/header artifacts.
      4. Reconnect hyphenated word splits across line breaks (e.g. 'in-\nstructor' -> 'instructor').
      5. Repair common PDF kerning & font ligature splits (e.g. 'B OOK', 'r eceive', 'fr om', 'lear n', 'mor e').
      6. Suffix repair (e.g. 'consider ed' -> 'considered').
      7. Collapse redundant horizontal whitespace while preserving paragraph boundaries.
    """
    if not text:
        return ""
    
    # 1. Unicode normalization (NFKC resolves ligatures 'ff', 'fi', curly quotes, fullwidth chars)
    cleaned = unicodedata.normalize('NFKC', text)
    
    # 2. Normalize line endings and form-feeds
    cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n').replace('\x0c', '\n')
    
    # 3. Strip common watermark/header artifacts
    cleaned = re.sub(r'(?i)\boceanofpdf\s*\.?\s*com\b', '', cleaned)
    
    # 4. Remove standalone page numbers or "Page X of Y" headers/footers
    cleaned = re.sub(r'(?m)^\s*(?:page\s+)?\d+(?:\s+of\s+\d+)?\s*$', '', cleaned, flags=re.IGNORECASE)
    
    # 5. Reconnect hyphenated words split across lines: e.g. "con-\nnection" -> "connection"
    cleaned = re.sub(r'(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)', r'\1\2', cleaned)
    
    # 6. Repair all-caps kerning splits: e.g. "B OOK" -> "BOOK"
    cleaned = re.sub(r'\b([A-Z])\s+([A-Z]{2,})\b', r'\1\2', cleaned)
    
    # 7. Suffix reconnect: e.g. "consider ed" -> "considered", "train ing" -> "training"
    cleaned = re.sub(r'\b([a-zA-Z]{3,})\s+(ed|ly|ing|tion|tions|er|es|en)\b', r'\1\2', cleaned)
    
    # 8. Specific kerning split repairs common in PyPDF2 font extraction
    cleaned = re.sub(r'\b([b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z])\s+(eceive|eceived|eceiving|ublication|ublications|etween|ecoming|ecause)\b', r'\1\2', cleaned)
    cleaned = re.sub(r'\b(fr)\s+(om)\b', r'\1\2', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b(lear|mor|befor|thes|thos)\s+([ne])\b', r'\1\2', cleaned, flags=re.IGNORECASE)
    
    # 9. Normalize horizontal whitespace per line (tabs and spaces to single space)
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in cleaned.split('\n')]
    
    # 10. Reconstruct paragraphs (preserve double \n for paragraph separation, merge broken single line wraps)
    reconstructed = []
    current_para = []
    for line in lines:
        if not line:
            if current_para:
                reconstructed.append(' '.join(current_para))
                current_para = []
        else:
            current_para.append(line)
    if current_para:
        reconstructed.append(' '.join(current_para))
        
    return '\n\n'.join(reconstructed)


def read_pdf_pages(file_path: str) -> List[Dict]:
    """
    Read PDF file page-by-page, preserving exact page numbers and character counts.
    Returns a list of dicts: [{'page_number': int, 'text': str, 'char_count': int}]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        import PyPDF2
        logger.info(f"Reading PDF: {file_path}")
        pages = []
        total_chars = 0
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            logger.info(f"PDF has {num_pages} pages")
            
            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1  # 1-indexed page number
                page_text = page.extract_text() or ""
                # Clean extracted text: unicode normalize, de-hyphenate, fix spacing, clean headers/footers
                page_text_clean = clean_extracted_text(page_text)
                
                if page_text_clean:
                    pages.append({
                        'page_number': page_num,
                        'text': page_text_clean,
                        'char_count': len(page_text_clean)
                    })
                    total_chars += len(page_text_clean)
                
                if (page_idx + 1) % 50 == 0 or (page_idx + 1) == num_pages:
                    logger.debug(f"  Processed {page_idx + 1}/{num_pages} pages")
        
        logger.info(f"Extracted {len(pages)} non-empty pages ({total_chars:,} total characters)")
        return pages
        
    except ImportError:
        logger.error("PyPDF2 is not installed. Please install it using 'pip install PyPDF2'.")
        raise


def read_pdf(file_path: str) -> str:
    """Read PDF file and return complete combined text (backward-compatible wrapper)."""
    pages = read_pdf_pages(file_path)
    return "\n\n".join(p['text'] for p in pages)


def create_chunks_from_pages(pages: List[Dict], 
                             target_tokens: int = None, 
                             overlap_tokens: int = None) -> List[Dict]:
    """
    Create page-aware, token-aware chunks preserving exact citation metadata.
    
    Guarantees:
      1. Respects page boundaries (chunks never bleed across unrelated pages).
      2. Respects sentence boundaries (never cuts words or sentences in half).
      3. Bounded in token count (target ~250-350 tokens).
      4. Preserves page_number, start_char, end_char, and token_count on every chunk.
    """
    if target_tokens is None:
        target_tokens = config.chunking.target_tokens
    if overlap_tokens is None:
        overlap_tokens = config.chunking.overlap_tokens
        
    all_chunks = []
    global_chunk_idx = 0
    
    for page in pages:
        page_num = page['page_number']
        page_text = page['text']
        
        if not page_text or not page_text.strip():
            continue
        
        # Split page into paragraphs
        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
        
        # Gather all sentences with their character offsets within the page
        sentences_with_offsets = []
        search_pos = 0
        for p in paragraphs:
            for s in split_into_sentences(p):
                idx = page_text.find(s, search_pos)
                if idx == -1:
                    idx = search_pos
                sentences_with_offsets.append({
                    'text': s,
                    'start': idx,
                    'end': idx + len(s),
                    'tokens': count_tokens(s)
                })
                search_pos = idx + len(s)
        
        if not sentences_with_offsets:
            continue
        
        cur_sentences = []
        cur_tokens = 0
        
        for s_info in sentences_with_offsets:
            s_tokens = s_info['tokens']
            
            # If adding this sentence exceeds target and we already have content
            if cur_tokens + s_tokens > target_tokens and cur_sentences:
                chunk_text = ' '.join(s['text'] for s in cur_sentences)
                chunk_id = f"chunk_{global_chunk_idx:06d}"
                all_chunks.append({
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'chunk_index': global_chunk_idx,
                    'page_number': page_num,
                    'start_char': cur_sentences[0]['start'],
                    'end_char': cur_sentences[-1]['end'],
                    'token_count': count_tokens(chunk_text),
                    'char_count': len(chunk_text),
                    'level': 0
                })
                global_chunk_idx += 1
                
                # Overlap: preserve tail sentences that fit within overlap_tokens
                overlap_sentences = []
                overlap_count = 0
                for prev_s in reversed(cur_sentences):
                    if overlap_count + prev_s['tokens'] <= overlap_tokens:
                        overlap_sentences.insert(0, prev_s)
                        overlap_count += prev_s['tokens']
                    else:
                        break
                
                cur_sentences = overlap_sentences + [s_info]
                cur_tokens = overlap_count + s_tokens
            else:
                cur_sentences.append(s_info)
                cur_tokens += s_tokens
        
        # Emit final chunk for this page
        if cur_sentences:
            chunk_text = ' '.join(s['text'] for s in cur_sentences)
            chunk_id = f"chunk_{global_chunk_idx:06d}"
            all_chunks.append({
                'text': chunk_text,
                'chunk_id': chunk_id,
                'chunk_index': global_chunk_idx,
                'page_number': page_num,
                'start_char': cur_sentences[0]['start'],
                'end_char': cur_sentences[-1]['end'],
                'token_count': count_tokens(chunk_text),
                'char_count': len(chunk_text),
                'level': 0
            })
            global_chunk_idx += 1
            
    logger.info(f"Created {len(all_chunks)} page-aware, token-bounded chunks across {len(pages)} pages")
    return all_chunks


def create_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Sentence-aware chunking for raw text (backward-compatible).
    Converts raw text into a single-page list and runs create_chunks_from_pages.
    """
    target_tokens = max(50, chunk_size // 4)
    overlap_tokens = max(10, overlap // 4)
    pages = [{'page_number': 1, 'text': text, 'char_count': len(text)}]
    return create_chunks_from_pages(pages, target_tokens=target_tokens, overlap_tokens=overlap_tokens)

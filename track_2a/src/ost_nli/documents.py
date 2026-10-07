"""Page-local source passages and diagnostic reference alignment."""
import hashlib
import re
from pathlib import Path
from .data import validate_document, words


def pdf_document(path, source_url=None, max_chars=1200, overlap_chars=200):
    from pypdf import PdfReader
    path = Path(path)
    pdf = PdfReader(path)
    if pdf.is_encrypted:
        raise ValueError('Encrypted booklet is unsupported')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    passages = []
    for page_number, page in enumerate(pdf.pages, 1):
        text = page.extract_text() or ''
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                boundary = text.rfind('\n', start + max_chars // 2, end)
                if boundary > start:
                    end = boundary
            excerpt = text[start:end]
            if excerpt.strip():
                passages.append({'id': f'page-{page_number}-offset-{start}',
                    'paragraph_id': f'page-{page_number}-offset-{start}',
                    'page': page_number, 'text': excerpt, 'char_start': start,
                    'char_end': end, 'source_url': source_url, 'source_sha256': digest})
            if end == len(text):
                break
            start = max(start + 1, end - overlap_chars)
    if not passages:
        raise ValueError('Booklet has no selectable text; OCR is required')
    return validate_document({'document_id': 'pdf-' + digest + '-plain-v2', 'source_url': source_url,
        'source_sha256': digest, 'pages': len(pdf.pages), 'passages': passages,
        'document_scope': 'full_booklet',
        'processing': {'version': 'page-offset-plain-v2', 'max_chars': max_chars,
                       'overlap_chars': overlap_chars, 'ocr': False}})


def reference_ngrams(text, n=5):
    # PDF line-break hyphenation only affects diagnostic matching, never source text.
    tokens = words(re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text))
    return {tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))}


def reference_coverage(passages, reference):
    gold = reference_ngrams(reference)
    if not gold:
        return None
    observed = set().union(*(reference_ngrams(p['text']) for p in passages))
    return len(observed & gold) / len(gold)

"""Implement fixed-size and structure-aware chunking."""
import hashlib
import re
from .models import Chunk, DocumentPage


def _chunk(page: DocumentPage, start: int, end: int) -> Chunk:
    raw = page.text[start:end]
    start += len(raw) - len(raw.lstrip())
    end -= len(raw) - len(raw.rstrip())
    text = page.text[start:end]
    identity = f'{page.document_id}:{page.page}:{start}:{end}:{text}'
    return Chunk(hashlib.sha256(identity.encode()).hexdigest()[:24], page.document_id, page.source, page.page,
                 text, {**page.metadata, 'start_char': start, 'end_char': end})


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping windows and retain page metadata."""
    if type(size) is not int or type(overlap) is not int or size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require integer size > 0 and 0 <= overlap < size')
    result, start = [], 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        if page.text[start:end].strip():
            result.append(_chunk(page, start, end))
        if end == len(page.text):
            break
        start = end - overlap
    return result


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Pack complete paragraphs; split long ones at a sentence or word boundary."""
    if type(max_size) is not int or max_size <= 0:
        raise ValueError('max_size must be a positive integer')
    ends = [m.end() for m in re.finditer(r'\n\n', page.text)] + [len(page.text)]
    result, start = [], 0
    while start < len(page.text):
        limit = min(start + max_size, len(page.text))
        candidates = [end for end in ends if start < end <= limit]
        end = max(candidates) if candidates else limit
        if not candidates and limit < len(page.text):
            boundaries = list(re.finditer(r'(?<=[.!?])\s+', page.text[start:limit]))
            if boundaries:
                end = start + boundaries[-1].end()
            else:
                boundary = page.text.rfind(' ', start, limit)
                if boundary > start:
                    end = boundary + 1
        if page.text[start:end].strip():
            result.append(_chunk(page, start, end))
        start = end
    return result

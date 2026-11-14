"""Read and normalize source data, retaining physical PDF page numbers."""
import hashlib
import json
import logging
from logging.handlers import BufferingHandler
import re
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from pypdf import PdfReader
from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page; keep its page number and source.

    Detect and record pages with no text. OCR, when needed, is a separate step.
    """
    pages = []
    for number, page in enumerate(PdfReader(path).pages, 1):
        error = None
        capture = BufferingHandler(100)
        pdf_logger = logging.getLogger('pypdf')
        pdf_logger.addHandler(capture)
        try:
            raw = page.extract_text(extraction_mode='layout') or ''
        except (ValueError, RuntimeError) as exc:
            raw, error = '', f'{type(exc).__name__}: {exc}'
        finally:
            pdf_logger.removeHandler(capture)
        lines = raw.splitlines()
        lines = [line for line in lines if not re.fullmatch(r'\s*(?:NIST AI 100-1\s+AI RMF 1\.0|Page \d+)\s*', line)]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        # The GenAI profile uses isolated page numbers, not body/table numbers.
        if lines and lines[0].strip() == str(number - 4):
            lines.pop(0)
        if lines and lines[-1].strip() == str(number - 4):
            lines.pop()
        text = normalize_text('\n'.join(lines))
        pages.append(DocumentPage(document_id, source, number, text,
                                  {'raw_characters': len(raw), 'empty': not bool(text), 'extraction_error': error,
                                   'extraction_warnings': [item.getMessage() for item in capture.buffer]}))
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph boundaries."""
    text = unicodedata.normalize('NFKC', text).replace('\r\n', '\n').replace('\r', '\n').replace('\u00ad', '')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    text = re.sub(r'(?<=[A-Za-z])-\s*\n\s*(?=[a-z])', '', text)
    return '\n\n'.join(' '.join(block.split()) for block in re.split(r'\n\s*\n', text) if block.strip())


def download_file(url: str, path: Path, expected_sha256: str) -> None:
    """Fetch a pinned public data/model asset; never accept a changed snapshot."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or parsed.username or parsed.password:
        raise ValueError('Only public HTTPS assets without embedded credentials are supported')
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Day06-Student-Lab/1.0'}), timeout=30) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f'Asset checksum mismatch: {path.name}; review the source/version explicitly')
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def prepare_corpus(root: Path) -> list[DocumentPage]:
    """Apply the declared page selection, emit provenance and extraction logs."""
    manifest_path = root / 'data/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    selected, log = [], []
    for source in manifest['sources']:
        path = root / source['file']
        download_file(source['source_path_or_url'], path, source['sha256'])
        pages = load_pdf(path, source['source_id'], source['source_path_or_url'])
        wanted = source['selected_pages']
        if len(wanted) != len(set(wanted)) or any(type(n) is not int or not 1 <= n <= len(pages) for n in wanted):
            raise ValueError('Manifest page IDs are duplicated or out of range')
        source['physical_pages'] = len(pages)
        source['page_character_counts'] = {}
        for page in pages:
            used = page.page in wanted
            log.append({'document_id': page.document_id, 'page': page.page, 'selected': used,
                        'characters': len(page.text), **page.metadata})
            if used:
                page.metadata.update({'year': source['year'], 'document_type': source['document_type']})
                selected.append(page)
                source['page_character_counts'][str(page.page)] = len(page.text)
    (root / 'results').mkdir(exist_ok=True)
    (root / 'results/preparation.json').write_text(json.dumps(log, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if len(selected) < 30 or any(not p.text or p.metadata['extraction_error'] for p in selected):
        raise ValueError('Need at least 30 readable content pages; inspect results/preparation.json')
    manifest['content_page_count'] = len(selected)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return selected

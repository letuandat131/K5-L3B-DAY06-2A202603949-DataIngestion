from pathlib import Path
import math

import pytest

from starter.src import chunking, evaluation, ingestion, store
from starter.src.models import Chunk, DocumentPage


def page(text):
    return DocumentPage('doc', 'https://example.org/doc.pdf', 7, text, {'year': 2023})


def test_normalization_keeps_paragraphs_and_unicode():
    assert ingestion.normalize_text('  cafe\u0301  ﬁle\nwrapped\n\n Next\t paragraph  ') == 'café file wrapped\n\nNext paragraph'


def test_empty_pdf_page_keeps_physical_page_and_error_metadata(tmp_path):
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    path = tmp_path / 'blank.pdf'
    writer.write(path)
    pages = ingestion.load_pdf(path, 'blank', 'https://example.org/blank.pdf')
    assert len(pages) == 1 and pages[0].page == 1
    assert pages[0].metadata['empty'] is True


def test_fixed_windows_cover_text_and_keep_stable_source_metadata():
    original = page('abcdefghij' * 8)
    chunks = chunking.fixed_size(original, 25, 5)
    assert chunks[0].text == original.text[:25]
    assert chunks[-1].metadata['end_char'] == len(original.text)
    assert all(c.page == 7 and c.metadata['year'] == 2023 for c in chunks)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunking.fixed_size(original, 25, 5)]


@pytest.mark.parametrize('size,overlap', [(0, 0), (10, 10), (10, -1)])
def test_bad_chunk_window_is_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunking.fixed_size(page('sample text'), size, overlap)


def test_paragraph_chunks_do_not_lose_text_and_preserve_short_paragraph():
    text = 'First paragraph is intact.\n\nSecond paragraph is also intact.\n\n' + 'long sentence ' * 12
    chunks = chunking.by_paragraph(page(text), 70)
    assert chunks and all(len(c.text) <= 70 for c in chunks)
    assert ''.join(''.join(c.text.split()) for c in chunks) == ''.join(text.split())
    assert 'First paragraph is intact.' in chunks[0].text
    for c in chunks:
        assert text[c.metadata['start_char']:c.metadata['end_char']] == c.text


def hit(text, doc='doc', number=7):
    return {'chunk_id': 'c1', 'text': text, 'score': .8,
            'metadata': {'document_id': doc, 'source': 'https://example.org/doc.pdf', 'page': number, 'year': 2023}}


QUESTION = {'id': 'q1', 'split': 'dev', 'question': 'What is required?',
            'gold_source_id': 'doc', 'gold_page': 7, 'evidence_quote': 'consent is required before publication',
            'expected_answer': 'Consent before publication.'}


def test_same_page_without_full_evidence_is_not_a_hit():
    result = evaluation.evaluate_query(QUESTION, [hit('consent is required')])
    assert result['hit1'] is False and result['hit3'] is False


def test_right_evidence_on_wrong_page_is_not_a_hit():
    assert evaluation.evaluate_query(QUESTION, [hit(QUESTION['evidence_quote'], number=8)])['hit3'] is False


def test_any_documented_alternative_evidence_can_match():
    question = {**QUESTION, 'alternative_evidence': [{'source_id': 'other', 'page': 2, 'quote': 'publication requires consent'}]}
    result = evaluation.evaluate_query(question, [hit('unrelated'), hit('Publication requires consent.', 'other', 2)])
    assert result['hit1'] is False and result['hit3'] is True and result['matched_ranks'] == [2]


def test_summary_uses_separate_split_denominators():
    traces = [{'strategy': 'A', 'split': 'dev', 'hit1': True, 'hit3': True},
              {'strategy': 'A', 'split': 'dev', 'hit1': False, 'hit3': True},
              {'strategy': 'A', 'split': 'test', 'hit1': False, 'hit3': False}]
    summary = evaluation.summarize(traces)
    assert summary['A']['dev']['hit1'] == {'n': 1, 'N': 2, 'rate': .5}
    assert summary['A']['test']['hit3'] == {'n': 0, 'N': 1, 'rate': 0.0}


def test_store_reload_filter_and_changed_source_cleanup(tmp_path):
    assert hasattr(store, 'QdrantStore'), 'Qdrant persistence adapter is not implemented yet'
    backend = store.QdrantStore(str(tmp_path / 'db'), 3)
    backend.create_schema()
    first = Chunk('old', 'doc', 'source', 1, 'old wording', {'year': 2023})
    second = Chunk('other', 'other', 'source2', 2, 'other document', {'year': 2024})
    backend.upsert('A', [first, second], [[1., 0., 0.], [0., 1., 0.]])
    backend.upsert('A', [first, second], [[1., 0., 0.], [0., 1., 0.]])
    assert backend.count('A') == 2
    updated = Chunk('new', 'doc', 'source', 1, 'updated wording', {'year': 2023})
    backend.upsert('A', [updated, second], [[1., 0., 0.], [0., 1., 0.]])
    backend.close()
    backend = store.QdrantStore(str(tmp_path / 'db'), 3)
    try:
        results = backend.search('A', [1., 0., 0.], filters={'year': 2023})
        assert backend.count('A') == 2 and len(results) == 1
        assert results[0]['chunk_id'] == 'new' and results[0]['text'] == 'updated wording'
        with pytest.raises(ValueError):
            backend.search('A', [1., 0., 0.], filters={'gold_source_id': 'doc'})
    finally:
        backend.close()


def test_bad_vectors_do_not_delete_existing_records(tmp_path):
    assert hasattr(store, 'QdrantStore'), 'Qdrant persistence adapter is not implemented yet'
    backend = store.QdrantStore(str(tmp_path / 'db'), 3)
    backend.create_schema()
    item = Chunk('one', 'doc', 'source', 1, 'valid text', {'year': 2023})
    try:
        backend.upsert('A', [item], [[1., 0., 0.]])
        with pytest.raises(ValueError):
            backend.upsert('A', [item], [[math.nan, 0., 0.]])
        with pytest.raises(ValueError):
            backend.upsert('A', [item], [[1., 0.]])
        assert backend.count('A') == 1
    finally:
        backend.close()


def test_freeze_hash_ignores_json_key_order_and_line_endings(tmp_path):
    from starter.src.run import fingerprint
    first, second = tmp_path / 'a.json', tmp_path / 'b.json'
    first.write_bytes(b'{"one": 1,\n"two": 2}\n')
    second.write_bytes(b'{"two": 2,\r\n"one": 1}\r\n')
    assert fingerprint(first) == fingerprint(second)


def test_B_and_C_share_chunk_ids_and_reject_boundary_changes():
    from starter.src.run import create_chunks
    config = {'strategies': {'A': {'chunking': 'fixed', 'max_chars': 70, 'overlap': 0, 'use_filter': False},
                             'B': {'chunking': 'paragraph', 'max_chars': 70, 'overlap': 0, 'use_filter': False},
                             'C': {'chunking': 'paragraph', 'max_chars': 70, 'overlap': 0, 'use_filter': True}}}
    chunks = create_chunks([page('A complete sentence.\n\nAnother complete sentence. ' * 4)], config)
    assert [c.chunk_id for c in chunks['B']] == [c.chunk_id for c in chunks['C']]
    config['strategies']['C']['max_chars'] = 90
    with pytest.raises(ValueError):
        create_chunks([page('example')], config)

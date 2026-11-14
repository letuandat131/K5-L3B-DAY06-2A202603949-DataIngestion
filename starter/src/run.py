"""Reproducible local pipeline: freeze evidence, select on dev, then evaluate test."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import normalize_text, prepare_corpus
from .store import QdrantStore

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ['data/manifest.json', 'data/normalized_pages.jsonl', 'configs/experiment.json',
          'benchmark/questions.json', 'benchmark/coherence_anchors.json']


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n' for item in values), encoding='utf-8')


def fingerprint(path):
    """Canonical data hashes survive Git CRLF/LF conversion across operating systems."""
    text = path.read_text(encoding='utf-8')
    value = [json.loads(line) for line in text.splitlines() if line] if path.suffix == '.jsonl' else json.loads(text)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def validate_benchmark(questions, pages):
    if len(questions) < 20 or len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Need at least 20 questions with unique IDs')
    if sum(q['split'] == 'dev' for q in questions) < 15 or sum(q['split'] == 'test' for q in questions) < 5:
        raise ValueError('Need at least 15 dev and 5 test questions')
    if any(q['split'] not in ('dev', 'test') for q in questions):
        raise ValueError('Unknown benchmark split')
    filtered = [q for q in questions if q.get('filter')]
    if len(filtered) < 3 or not any(q['split'] == 'test' for q in filtered):
        raise ValueError('Need three metadata questions including a test question')
    lookup = {(p.document_id, p.page): p for p in pages}
    compact = lambda text: ' '.join(normalize_text(text).casefold().split())
    for q in questions:
        if not q['question'].strip() or not q['expected_answer'].strip():
            raise ValueError('Question and expected answer must be non-empty')
        for key, value in q.get('filter', {}).items():
            if key != 'year' or type(value) is not int or str(value) not in q['question']:
                raise ValueError('This benchmark only uses publication years explicitly stated in the question')
        golds = [{'source_id': q['gold_source_id'], 'page': q['gold_page'], 'quote': q['evidence_quote']},
                 *q.get('alternative_evidence', [])]
        for gold in golds:
            source = lookup.get((gold['source_id'], gold['page']))
            if source is None or not compact(gold['quote']) or compact(gold['quote']) not in compact(source.text):
                raise ValueError(f"Gold evidence missing from selected physical source page: {q['id']}")
            if q.get('filter') and any(source.metadata[k] != v for k, v in q['filter'].items()):
                raise ValueError('Gold alternative contradicts the question-supplied filter')


def create_chunks(pages, config):
    settings = config['strategies']
    if set(settings) != {'A', 'B', 'C'} or settings['A']['max_chars'] != settings['B']['max_chars']:
        raise ValueError('A/B must share the declared character limit')
    if any(s['overlap'] != 0 for s in settings.values()):
        raise ValueError('This controlled experiment fixes overlap to zero')
    if {k: v for k, v in settings['B'].items() if k != 'use_filter'} != {k: v for k, v in settings['C'].items() if k != 'use_filter'}:
        raise ValueError('B/C may differ only in filter application')
    chunks = {
        'A': [c for p in pages for c in fixed_size(p, settings['A']['max_chars'], 0)],
        'B': [c for p in pages for c in by_paragraph(p, settings['B']['max_chars'])],
    }
    chunks['C'] = chunks['B']
    return chunks


def audit_inputs(root, chunks, questions, traces):
    coherence = []
    anchors = json.loads((root / 'benchmark/coherence_anchors.json').read_text(encoding='utf-8'))
    for name, items in chunks.items():
        for anchor in anchors:
            matches = [c for c in items if c.document_id == anchor['document_id'] and c.page == anchor['page']
                       and c.metadata['start_char'] <= anchor['position'] < c.metadata['end_char']]
            if len(matches) != 1:
                raise ValueError('Fixed source anchor does not have exactly one covering chunk')
            coherence.append({'strategy': name, 'anchor': anchor, 'chunk': asdict(matches[0]), 'score': None, 'reason': None})
    return {'coherence': coherence, 'grounding': [t for t in traces if t['split'] == 'test'],
            'review_method': 'Inspect source/chunks; do not derive coherence or grounding labels from Hit@3',
            'student_reviewed': False}


def main() -> None:
    """Run the pipeline and preserve the five evaluation dimensions.

    1. Load the corpus and benchmark; validate at least 30 content pages.
    2. Build strategies A (fixed), B (paragraph) and C (B + metadata filter).
    3. Embed with the same model and store vectors/metadata per strategy.
    4. Run the same dev/test questions and save full top-3 traces.
    5. Report retrieval metrics and prepare samples for manual assessment.

    Follow the five dimensions in ../../BENCHMARK.md.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare-only', action='store_true')
    action.add_argument('--freeze-benchmark', action='store_true')
    parser.add_argument('--output', default='results/benchmark')
    args = parser.parse_args()
    output = ROOT / args.output
    if not args.prepare_only and not args.freeze_benchmark and (output / 'traces.jsonl').exists():
        raise FileExistsError('Results already exist; use a new --output directory to preserve them')
    started, cpu_started = time.perf_counter(), time.process_time()
    started_at = datetime.now(timezone.utc).isoformat()
    pages = prepare_corpus(ROOT)
    write_jsonl(ROOT / 'data/normalized_pages.jsonl', [asdict(p) for p in pages])
    if args.prepare_only:
        print(f'Prepared {len(pages)} content pages; no retrieval or model API calls.')
        return
    config = json.loads((ROOT / 'configs/experiment.json').read_text(encoding='utf-8'))
    questions = json.loads((ROOT / 'benchmark/questions.json').read_text(encoding='utf-8'))
    validate_benchmark(questions, pages)
    hashes = {name: fingerprint(ROOT / name) for name in INPUTS}
    freeze_path = ROOT / 'benchmark/freeze.json'
    if args.freeze_benchmark:
        if freeze_path.exists():
            raise FileExistsError('Benchmark already frozen; do not overwrite it after inspecting results')
        if list((ROOT / 'results').glob('*/traces.jsonl')):
            raise ValueError('Retrieval traces already exist; cannot claim this is a fresh holdout freeze')
        write_json(freeze_path, {'created_at': datetime.now(timezone.utc).isoformat(), 'hashes': hashes,
                                'dev_ids': [q['id'] for q in questions if q['split'] == 'dev'],
                                'test_ids': [q['id'] for q in questions if q['split'] == 'test'],
                                'selection_policy': config['selection_policy'], 'corpus_retrieval_runs_before_freeze': 0})
        print('Benchmark, gold, source anchors and configurations frozen before retrieval.')
        return
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    if hashes != freeze['hashes']:
        raise ValueError('Frozen corpus/benchmark/configuration changed; do not silently reuse holdout claims')
    output.mkdir(parents=True, exist_ok=True)
    chunks = create_chunks(pages, config)
    for name, items in chunks.items():
        write_jsonl(output / f'chunks_{name}.jsonl', [asdict(c) for c in items])
    embedder = MyEmbedder(config['model']['name'])
    cache_initial_entries = len(embedder.cache)
    backend = QdrantStore(str(ROOT / config['store']['path']), embedder.dimensions)
    stats, vectors, persistence, traces = {}, {}, {}, []
    selection = None
    try:
        backend.create_schema()
        for name, items in chunks.items():
            before = embedder.usage.copy()
            vectors[name] = embedder.embed_documents([c.text for c in items])
            tick = time.perf_counter()
            backend.upsert(name, items, vectors[name])
            durations = {'index_seconds': time.perf_counter() - tick,
                         'embedding_seconds': embedder.usage['inference_seconds'] - before['inference_seconds']}
            count = backend.count(name)
            if count != len(items):
                raise ValueError('Index count differs from the declared chunks')
            stats[name] = {'chunk_count': count, 'min_chars': min(len(c.text) for c in items),
                           'max_chars': max(len(c.text) for c in items), 'mean_chars': sum(len(c.text) for c in items) / count,
                           'total_chars': sum(len(c.text) for c in items),
                           'newly_embedded_texts': embedder.usage['computed_texts'] - before['computed_texts'], **durations}
            backend.upsert(name, items, vectors[name])
            persistence[name] = {'count_before_reinsert': count, 'count_after_reinsert': backend.count(name)}
        backend.close()
        backend = QdrantStore(str(ROOT / config['store']['path']), embedder.dimensions)
        for name in chunks:
            persistence[name]['count_after_reopen'] = backend.count(name)
            if len(set(persistence[name].values())) != 1:
                raise ValueError('Persistence/idempotence check failed')
        write_json(output / 'persistence.json', persistence)
        for split in ('dev', 'test'):
            if split == 'test':
                metrics = summarize(traces)
                selected = min(chunks, key=lambda name: (-metrics[name]['dev']['hit3']['rate'],
                                                        -metrics[name]['dev']['hit1']['rate'],
                                                        len(chunks[name]), name))
                selection = {'selected_strategy': selected, 'policy': config['selection_policy'],
                             'frozen_at': datetime.now(timezone.utc).isoformat(), 'dev_metrics': metrics,
                             'test_queries_run_at_selection': 0, 'input_hashes': hashes}
                write_json(output / 'selection.json', selection)
            for name in chunks:
                for q in (q for q in questions if q['split'] == split):
                    query_vector = embedder.embed_query(q['question'])
                    filters = q.get('filter') if config['strategies'][name]['use_filter'] else None
                    tick = time.perf_counter()
                    hits = backend.search(name, query_vector, top_k=config['top_k'], filters=filters)
                    trace = {'strategy': name, **evaluate_query(q, hits), 'filter_applied': bool(filters),
                             'search_seconds': time.perf_counter() - tick, 'evaluated_at': datetime.now(timezone.utc).isoformat()}
                    traces.append(trace)
                    write_jsonl(output / 'traces.jsonl', traces)
            print(f'{split}: {sum(t["split"] == split for t in traces)} queries across A/B/C', flush=True)
        write_json(output / 'metrics.json', summarize(traces))
        write_json(output / 'manual_audit_inputs.json', audit_inputs(ROOT, chunks, questions, traces))
    finally:
        backend.close()
    manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))
    write_json(output / 'run_info.json', {'started_at': started_at, 'finished_at': datetime.now(timezone.utc).isoformat(),
               'wall_seconds': time.perf_counter() - started, 'cpu_seconds': time.process_time() - cpu_started,
               'content_pages': len(pages), 'physical_pages_read': sum(s['physical_pages'] for s in manifest['sources']),
               'model': config['model'], 'store': config['store'], 'strategy_stats': stats,
               'embedding_usage': embedder.usage, 'cache_initial_entries': cache_initial_entries,
               'selected_on_dev': selection['selected_strategy'], 'input_hashes': hashes,
               'python': sys.version, 'platform': platform.platform(),
               'dependencies': {p: importlib.metadata.version(p) for p in ['numpy', 'onnxruntime', 'tokenizers', 'pypdf', 'qdrant-client']},
               'cost_notes': 'Local CPU inference, USD 0 paid API; initial dependency/model downloads excluded from run wall time. B/C share chunks and cached embeddings; ranking timings exclude query embedding.'})
    print(json.dumps({'selected_on_dev': selection['selected_strategy'], 'metrics': summarize(traces),
                      'wall_seconds': round(time.perf_counter() - started, 3), 'paid_api_calls': 0}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        write_json(ROOT / 'results/last_error.json', {'type': type(exc).__name__, 'message': str(exc)[:500],
                                                    'at': datetime.now(timezone.utc).isoformat(), 'fallback_used': False})
        raise

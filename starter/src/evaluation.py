"""Evaluate complete retrieved evidence, separately from reviewer audit labels."""
from typing import Any
from .ingestion import normalize_text


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check source/page and complete gold evidence; keep hit@1 and hit@3 trace.

    Accept any documented alternative gold evidence. Do not infer human grounding
    or chunk coherence labels from exact string match.
    """
    gold = [{'source_id': question['gold_source_id'], 'page': question['gold_page'], 'quote': question['evidence_quote']},
            *question.get('alternative_evidence', [])]
    compact = lambda text: ' '.join(normalize_text(text).casefold().split())
    matched = []
    for rank, hit in enumerate(hits[:3], 1):
        metadata = hit['metadata']
        if any(metadata['document_id'] == item['source_id'] and metadata['page'] == item['page']
               and compact(item['quote']) and compact(item['quote']) in compact(hit['text']) for item in gold):
            matched.append(rank)
    return {'id': question['id'], 'split': question['split'], 'question': question['question'],
            'expected_answer': question['expected_answer'], 'gold': gold, 'filter': question.get('filter'),
            'hits': hits[:3], 'matched_ranks': matched, 'hit1': 1 in matched, 'hit3': bool(matched)}


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 with separate split denominators."""
    result = {}
    for strategy in sorted({t['strategy'] for t in traces}):
        result[strategy] = {}
        for split in ('dev', 'test'):
            group = [t for t in traces if t['strategy'] == strategy and t['split'] == split]
            if group:
                result[strategy][split] = {key: {'n': sum(bool(t[key]) for t in group), 'N': len(group),
                                               'rate': sum(bool(t[key]) for t in group) / len(group)}
                                          for key in ('hit1', 'hit3')}
    return result

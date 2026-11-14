"""Choose a real embedding provider (API or local) and implement this contract."""
import hashlib
import json
import time
from pathlib import Path
from typing import Protocol
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from .ingestion import download_file


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Pinned MiniLM ONNX adapter: masked mean pooling, then L2 normalization."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        root = Path(__file__).resolve().parents[2]
        config = json.loads((root / 'configs/experiment.json').read_text(encoding='utf-8'))['model']
        if model_name != config['name']:
            raise ValueError('Only the declared frozen embedding model is supported')
        self.config, self.dimensions = config, config['dimensions']
        directory = root / '.cache' / ('model-' + config['revision'][:8])
        for name, checksum in config['files'].items():
            url = f"https://huggingface.co/{model_name}/resolve/{config['revision']}/{name}"
            download_file(url, directory / Path(name).name, checksum)
        options = ort.SessionOptions()
        options.intra_op_num_threads = config['threads']
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / 'model_quantized.onnx'), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        self.cache_path = directory / 'embeddings.json'
        self.fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        saved = json.loads(self.cache_path.read_text(encoding='utf-8')) if self.cache_path.exists() else {}
        if saved and saved['fingerprint'] != self.fingerprint:
            raise ValueError('Embedding cache settings differ from the frozen model configuration')
        self.cache = saved.get('vectors', {})
        self.usage = {'computed_texts': 0, 'cache_hits': 0, 'input_tokens': 0, 'truncated_texts': 0,
                      'inference_seconds': 0.0, 'paid_api_calls': 0, 'cost_usd': 0.0}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError('Embedding input must contain non-empty text strings')
        keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        missing = {key: text for key, text in zip(keys, texts) if key not in self.cache}
        self.usage['cache_hits'] += len(texts) - len(missing)
        items = list(missing.items())
        for start in range(0, len(items), self.config['batch_size']):
            batch = items[start:start + self.config['batch_size']]
            self.tokenizer.no_truncation()
            full = self.tokenizer.encode_batch([text for _, text in batch])
            lengths = [sum(item.attention_mask) for item in full]
            self.usage['input_tokens'] += sum(lengths)
            self.usage['truncated_texts'] += sum(n > self.config['max_tokens'] for n in lengths)
            self.tokenizer.enable_truncation(max_length=self.config['max_tokens'])
            encoded = self.tokenizer.encode_batch([text for _, text in batch])
            feed = {'input_ids': np.array([e.ids for e in encoded], dtype=np.int64),
                    'attention_mask': np.array([e.attention_mask for e in encoded], dtype=np.int64),
                    'token_type_ids': np.array([e.type_ids for e in encoded], dtype=np.int64)}
            tick = time.perf_counter()
            output = self.session.run(None, {item.name: feed[item.name] for item in self.session.get_inputs()})[0]
            mask = feed['attention_mask'][..., None]
            pooled = (output * mask).sum(axis=1) / mask.sum(axis=1)
            vectors = (pooled / np.linalg.norm(pooled, axis=1, keepdims=True)).astype(np.float32)
            if vectors.shape != (len(batch), self.dimensions) or not np.isfinite(vectors).all():
                raise ValueError('Model returned an invalid embedding tensor')
            self.usage['inference_seconds'] += time.perf_counter() - tick
            self.usage['computed_texts'] += len(batch)
            for (key, _), vector in zip(batch, vectors):
                self.cache[key] = vector.tolist()
        if missing:
            temporary = self.cache_path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'fingerprint': self.fingerprint, 'vectors': self.cache}), encoding='utf-8')
            temporary.replace(self.cache_path)
        return [self.cache[key] for key in keys]

    def embed_query(self, question: str) -> list[float]:
        return self.embed_documents([question])[0]

"""Cross-model, cross-dataset chunking study.

Question: does chunking improve offline transcription accuracy, and does
the answer depend on model architecture?
"""

from __future__ import annotations
import argparse, gc, json, os, re, time, unicodedata
from typing import Optional

import numpy as np
import psutil
import soundfile as sf
import jiwer

PER_CLIP_LIMIT = 150
SR = 16000

TASHKEEL = re.compile(r'[ً-ٰۖ-ۭ]')


def norm(t):
    t = unicodedata.normalize('NFC', t or '')
    t = TASHKEEL.sub('', t)
    return re.sub(r'\s+', ' ', t).strip()


def detok_sp(text):
    if '▁' in text:
        return text.replace(' ', '').replace('▁', ' ').strip()
    return text


def to16k(s, sr):
    if s.ndim > 1:
        s = s.mean(axis=1)
    if sr != SR:
        ratio = SR / sr
        idx = np.linspace(0, len(s) - 1, int(len(s) * ratio)).astype('int')
        s = s[idx].astype('float32')
        sr = SR
    return s, sr


def boundary_dedup(prev, new, max_n=5):
    if not prev or not new:
        return new
    n_max = min(max_n, len(prev), len(new))
    for n in range(n_max, 0, -1):
        if prev[-n:] == new[:n]:
            return new[n:]
    return new


class Backend:
    name: str
    rss_load_mib: float = 0.0
    def transcribe_chunk(self, samples, sr):
        raise NotImplementedError


class SherpaRNNT(Backend):
    name = 'nemo-fastconformer-ar-pcd'

    def __init__(self, model_dir):
        import sherpa_onnx
        self.r = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f'{model_dir}/encoder.onnx',
            decoder=f'{model_dir}/decoder.onnx',
            joiner=f'{model_dir}/joiner.onnx',
            tokens=f'{model_dir}/tokens.txt',
            num_threads=4, decoding_method='greedy_search',
            model_type='nemo_transducer')

    def transcribe_chunk(self, samples, sr):
        s = self.r.create_stream()
        t0 = time.perf_counter()
        s.accept_waveform(sr, samples.copy())
        self.r.decode_stream(s)
        dt = time.perf_counter() - t0
        return norm(s.result.text), dt


class WhisperHF(Backend):
    def __init__(self, model_id, name):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import torch
        torch.set_num_threads(4)
        self.torch = torch
        self.name = name
        self.processor = WhisperProcessor.from_pretrained(model_id)
        m = WhisperForConditionalGeneration.from_pretrained(model_id)
        getattr(m, 'eval')()
        self.model = m
        if hasattr(self.model.generation_config, 'language'):
            self.model.generation_config.language = 'arabic'
        if hasattr(self.model.generation_config, 'task'):
            self.model.generation_config.task = 'transcribe'

    def transcribe_chunk(self, samples, sr):
        inputs = self.processor(samples, sampling_rate=sr, return_tensors='pt')
        t0 = time.perf_counter()
        with self.torch.no_grad():
            ids = self.model.generate(
                inputs.input_features,
                num_beams=1, do_sample=False, max_new_tokens=256,
            )
        dt = time.perf_counter() - t0
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return norm(text), dt


def run_config(backend, clips, chunk_ms, overlap_ms, proc, dataset_name):
    rss_peak = proc.memory_info().rss / 1024 / 1024
    refs, hyps = [], []
    audio_s = decode_s = 0.0
    n_words_ref = 0
    per_subgroup = {}
    canonical_prefix_hallucinations = 0

    for c in clips:
        samples = c['samples']
        if chunk_ms is None:
            chunks = [samples]
        else:
            chunk_samples = int(chunk_ms * SR / 1000)
            overlap_samples = int(overlap_ms * SR / 1000)
            step = max(1, chunk_samples - overlap_samples)
            chunks = []
            for start in range(0, len(samples), step):
                chunk = samples[start:start + chunk_samples]
                if len(chunk) < SR * 0.1:
                    continue
                chunks.append(chunk)
                if start + chunk_samples >= len(samples):
                    break

        all_words = []; last = []
        for chunk in chunks:
            try:
                text, dt = backend.transcribe_chunk(chunk, SR)
                decode_s += dt
            except Exception as e:
                continue
            if not text:
                continue
            words = [w for w in text.split() if w]
            if chunk_ms is not None and overlap_ms > 0:
                words = boundary_dedup(last, words)
            all_words.extend(words)
            last = words

        hyp = ' '.join(all_words)
        audio_s += len(samples) / SR
        refs.append(c['ref']); hyps.append(hyp)
        n_words_ref += len(c['ref'].split())
        per_subgroup.setdefault(c['subgroup'], []).append((c['ref'], hyp))
        if hyp.startswith('بسم') and not c['ref'].startswith('بسم'):
            canonical_prefix_hallucinations += 1
        rss_peak = max(rss_peak, proc.memory_info().rss / 1024 / 1024)

    overall = jiwer.wer(refs, hyps) if refs else None
    per = {r: jiwer.wer([x[0] for x in v], [x[1] for x in v])
           for r, v in per_subgroup.items() if v}
    return {
        'backend': backend.name,
        'dataset': dataset_name,
        'chunk_ms': chunk_ms,
        'overlap_ms': overlap_ms,
        'overall_wer': overall,
        'audio_seconds': audio_s,
        'decode_seconds': decode_s,
        'rtf': decode_s / audio_s if audio_s > 0 else None,
        'words_per_sec': n_words_ref / decode_s if decode_s > 0 else None,
        'n_clips': len(refs),
        'n_words_ref': n_words_ref,
        'per_subgroup_wer': per,
        'canonical_prefix_hallucinations': canonical_prefix_hallucinations,
        'rss_peak_mib': rss_peak,
        'rss_load_mib': backend.rss_load_mib,
        'first_text_latency_ms': chunk_ms,
    }


def cache_everyayah(token, max_clips=150):
    from datasets import load_dataset
    ds = load_dataset('tarteel-ai/everyayah', split='train', streaming=True, token=token)
    clips = []; counts = {}
    PER_RECITER = max_clips // 3; MAX_RECITERS = 3
    for row in ds:
        rec = (row.get('reciter') or '').lower()
        if not rec: continue
        if counts.get(rec, 0) >= PER_RECITER: continue
        if rec not in counts and len(counts) >= MAX_RECITERS: continue
        audio = row.get('audio')
        if audio is None: continue
        sa = audio.get_all_samples()
        arr = sa.data.numpy()
        if arr.ndim > 1: arr = arr.mean(axis=0)
        arr = np.asarray(arr, dtype='float32')
        samples, sr = to16k(arr, sa.sample_rate)
        counts[rec] = counts.get(rec, 0) + 1
        clips.append({'samples': samples, 'sr': sr, 'subgroup': rec,
                      'ref': norm(row.get('text', ''))})
        if all(c >= PER_RECITER for c in counts.values()) and len(counts) >= MAX_RECITERS:
            break
    print(f'cached everyayah {len(clips)} clips: {counts}', flush=True)
    return clips


def cache_fleurs(token, max_clips=150):
    from datasets import load_dataset
    ds = load_dataset('google/fleurs', 'ar_eg', split='test', streaming=True, token=token)
    clips = []
    for row in ds:
        if len(clips) >= max_clips: break
        audio = row.get('audio')
        if audio is None: continue
        sa = audio.get_all_samples()
        arr = sa.data.numpy()
        if arr.ndim > 1: arr = arr.mean(axis=0)
        arr = np.asarray(arr, dtype='float32')
        samples, sr = to16k(arr, sa.sample_rate)
        clips.append({'samples': samples, 'sr': sr, 'subgroup': 'fleurs-ar',
                      'ref': norm(row.get('transcription') or row.get('raw_transcription') or '')})
    print(f'cached fleurs {len(clips)} clips', flush=True)
    return clips


CHUNK_GRID = [None, 4000, 8000, 10000, 12000, 15000, 20000, 30000]
OVERLAP_GRID = [0, 500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/results/sweep.jsonl')
    ap.add_argument('--models', default='nemo,whisper-tiny,whisper-small,whisper-tiny-ar-quran,whisper-base-ar-quran')
    ap.add_argument('--datasets', default='everyayah,fleurs')
    ap.add_argument('--model-dir', default='/tmp/rnnt')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    proc = psutil.Process(os.getpid())

    datasets = {}
    if 'everyayah' in args.datasets:
        datasets['everyayah'] = cache_everyayah(os.environ['HF_TOKEN'])
    if 'fleurs' in args.datasets:
        try:
            datasets['fleurs'] = cache_fleurs(os.environ['HF_TOKEN'])
        except Exception as e:
            print(f'fleurs cache failed: {e}', flush=True)

    BUILDERS = {
        'nemo':                  lambda: SherpaRNNT(args.model_dir),
        'whisper-tiny':          lambda: WhisperHF('openai/whisper-tiny', 'whisper-tiny'),
        'whisper-small':         lambda: WhisperHF('openai/whisper-small', 'whisper-small'),
        'whisper-tiny-ar-quran': lambda: WhisperHF('tarteel-ai/whisper-tiny-ar-quran', 'whisper-tiny-ar-quran'),
        'whisper-base-ar-quran': lambda: WhisperHF('tarteel-ai/whisper-base-ar-quran', 'whisper-base-ar-quran'),
    }

    out_f = open(args.out, 'a')
    total = 0
    for model_key in args.models.split(','):
        model_key = model_key.strip()
        if model_key not in BUILDERS:
            print(f'unknown model: {model_key}', flush=True); continue
        print(f'\n=== loading {model_key} ===', flush=True)
        rss0 = proc.memory_info().rss / 1024 / 1024
        try:
            backend = BUILDERS[model_key]()
        except Exception as e:
            print(f'load failed: {e}', flush=True); continue
        backend.rss_load_mib = proc.memory_info().rss / 1024 / 1024 - rss0
        print(f'  loaded, +{backend.rss_load_mib:.0f} MiB', flush=True)

        for ds_name, clips in datasets.items():
            for chunk_ms in CHUNK_GRID:
                overlaps = [0] if chunk_ms is None else OVERLAP_GRID
                for ov in overlaps:
                    if chunk_ms is not None and ov >= chunk_ms: continue
                    total += 1
                    label = 'full' if chunk_ms is None else f'{chunk_ms}ms'
                    print(f'\n[{total}] {model_key} on {ds_name} chunk={label} ov={ov}ms', flush=True)
                    gc.collect()
                    res = run_config(backend, clips, chunk_ms, ov, proc, ds_name)
                    out_f.write(json.dumps(res, ensure_ascii=False) + '\n')
                    out_f.flush()
                    print(f'  -> WER={res["overall_wer"]:.4f} RTF={res["rtf"]:.3f} '
                          f'wps={res["words_per_sec"]:.1f} RSS={res["rss_peak_mib"]:.0f}MiB '
                          f'بسم={res["canonical_prefix_hallucinations"]}', flush=True)
        del backend
        gc.collect()

    out_f.close()
    print(f'\ndone -- {total} configs in {args.out}', flush=True)


if __name__ == '__main__':
    main()

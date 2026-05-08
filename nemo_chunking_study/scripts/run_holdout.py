"""Cross-dataset NeMo bench — top v2 strategies on a held-out (non-training) dataset.

This validates that the chunking trick is real and not data-contamination from
everyayah being in NeMo's training set.

Datasets supported:
  arabic-speech-corpus   (Nawar Halabi MSA, NOT in NeMo training)
  mgb3                    (Egyptian Arabic, NOT in NeMo training)
  cv-17-test              (Mozilla CV 17 ar test split)
"""
from __future__ import annotations
import argparse, gc, json, os, re, time, unicodedata
import numpy as np, psutil, soundfile as sf, jiwer

PER_CLIP = 100; SR = 16000
TASHKEEL = re.compile(r'[ً-ٰۖ-ۭ]')

def norm(t):
    t = unicodedata.normalize('NFC', t or '')
    t = TASHKEEL.sub('', t)
    return re.sub(r'\s+', ' ', t).strip()

def to16k(s, sr):
    if s.ndim > 1: s = s.mean(axis=1)
    if sr != SR:
        ratio = SR / sr
        idx = np.linspace(0, len(s)-1, int(len(s)*ratio)).astype('int')
        s = s[idx].astype('float32'); sr = SR
    return s, sr

def boundary_dedup(prev, new, max_n=5):
    if not prev or not new: return new
    n_max = min(max_n, len(prev), len(new))
    for n in range(n_max, 0, -1):
        if prev[-n:] == new[:n]:
            return new[n:]
    return new

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', required=True, choices=['arabic-speech-corpus', 'mgb3', 'cv-17-test', 'fleurs'])
ap.add_argument('--out', default='/tmp/results/holdout.jsonl')
args = ap.parse_args()

import sherpa_onnx
recog = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder='/tmp/rnnt/encoder.onnx',
    decoder='/tmp/rnnt/decoder.onnx',
    joiner='/tmp/rnnt/joiner.onnx',
    tokens='/tmp/rnnt/tokens.txt',
    num_threads=4, decoding_method='greedy_search',
    model_type='nemo_transducer')
print('NeMo loaded', flush=True)

def transcribe(samples):
    s = recog.create_stream()
    t0 = time.perf_counter()
    s.accept_waveform(SR, samples.copy())
    recog.decode_stream(s)
    dt = time.perf_counter() - t0
    return norm(s.result.text), dt

def chunks_fixed(samples, win_ms, ov_ms):
    win = int(win_ms * SR / 1000); ov = int(ov_ms * SR / 1000)
    step = max(1, win - ov)
    out = []
    for start in range(0, len(samples), step):
        c = samples[start:start+win]
        if len(c) < SR * 0.1: continue
        out.append(c)
        if start + win >= len(samples): break
    return out

# Top v2 strategies + baseline
strategies = [
    ('full', lambda s: [s]),
    ('fixed_11000_100', lambda s: chunks_fixed(s, 11000, 100)),
    ('fixed_10500_100', lambda s: chunks_fixed(s, 10500, 100)),
    ('fixed_11000_0_trim', lambda s: chunks_fixed(s, 11000, 0)),  # trim done at clip level
    ('fixed_10000_500', lambda s: chunks_fixed(s, 10000, 500)),
    ('fixed_8000_500', lambda s: chunks_fixed(s, 8000, 500)),
    ('fixed_4000_0', lambda s: chunks_fixed(s, 4000, 0)),
]

# Cache dataset
print(f'--- cache {args.dataset} ---', flush=True)
clips = []
from datasets import load_dataset
HF = os.environ.get('HF_TOKEN')

if args.dataset == 'arabic-speech-corpus':
    ds = load_dataset('arabic_speech_corpus', split='test', streaming=True, token=HF, trust_remote_code=True)
    for row in ds:
        if len(clips) >= PER_CLIP: break
        audio = row.get('audio')
        text = row.get('text') or row.get('transcription') or ''
        if not audio or not text: continue
        arr = np.asarray(audio['array'], dtype='float32')
        samples, sr = to16k(arr, audio['sampling_rate'])
        clips.append({'samples': samples, 'sr': sr, 'subgroup': 'asc', 'ref': norm(text)})
elif args.dataset == 'mgb3':
    ds = load_dataset('omarxadel/MGB3', split='test', streaming=True, token=HF, trust_remote_code=True)
    for row in ds:
        if len(clips) >= PER_CLIP: break
        audio = row.get('audio')
        text = row.get('text') or row.get('transcription') or row.get('sentence') or ''
        if not audio or not text: continue
        arr = np.asarray(audio['array'], dtype='float32')
        samples, sr = to16k(arr, audio['sampling_rate'])
        clips.append({'samples': samples, 'sr': sr, 'subgroup': 'mgb3', 'ref': norm(text)})
elif args.dataset == 'cv-17-test':
    ds = load_dataset('mozilla-foundation/common_voice_17_0', 'ar', split='test', streaming=True, token=HF, trust_remote_code=True)
    for row in ds:
        if len(clips) >= PER_CLIP: break
        audio = row.get('audio')
        text = row.get('sentence', '')
        if not audio or not text: continue
        arr = np.asarray(audio['array'], dtype='float32')
        samples, sr = to16k(arr, audio['sampling_rate'])
        clips.append({'samples': samples, 'sr': sr, 'subgroup': 'cv-ar', 'ref': norm(text)})
elif args.dataset == 'fleurs':
    ds = load_dataset('google/fleurs', 'ar_eg', split='test', streaming=True, token=HF, trust_remote_code=True)
    for row in ds:
        if len(clips) >= PER_CLIP: break
        audio = row.get('audio')
        text = row.get('transcription') or row.get('raw_transcription', '')
        if not audio or not text: continue
        arr = np.asarray(audio['array'], dtype='float32')
        samples, sr = to16k(arr, audio['sampling_rate'])
        clips.append({'samples': samples, 'sr': sr, 'subgroup': 'fleurs-ar', 'ref': norm(text)})

print(f'cached: {len(clips)} clips', flush=True)
if not clips:
    print('NO CLIPS — aborting'); exit(1)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
out_f = open(args.out, 'a')

for name, fn in strategies:
    print(f'\n[{name}]', flush=True)
    refs, hyps = [], []
    audio_s = decode_s = 0
    for c in clips:
        ws = []; last = []
        for chunk in fn(c['samples']):
            try:
                t, dt = transcribe(chunk)
                decode_s += dt
                if not t: continue
                w = [x for x in t.split() if x]
                w = boundary_dedup(last, w)
                ws.extend(w); last = w
            except Exception as e:
                print(f'  chunk err: {e}'); continue
        hyp = ' '.join(ws)
        refs.append(c['ref']); hyps.append(hyp)
        audio_s += len(c['samples']) / SR
    wer = jiwer.wer(refs, hyps)
    rtf = decode_s / audio_s if audio_s > 0 else None
    res = {'dataset': args.dataset, 'strategy': name, 'overall_wer': wer,
           'audio_seconds': audio_s, 'decode_seconds': decode_s, 'rtf': rtf,
           'n_clips': len(refs)}
    out_f.write(json.dumps(res, ensure_ascii=False) + '\n'); out_f.flush()
    print(f'  WER={wer*100:.2f}%  RTF={rtf:.4f}', flush=True)

out_f.close()
print('=== DONE ===', flush=True)

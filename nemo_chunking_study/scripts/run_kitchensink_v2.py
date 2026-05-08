"""GOAT chunking sweep on NeMo, on held-out + training-overlap datasets.

Datasets:
  everyayah       (in NeMo's training data — used as in-distribution baseline)
  common_voice_17 (held-out — never seen during training, validates the
                   chunking effect is real and not data-overlap)

For each dataset we run the same kitchen-sink strategy grid. If chunking
is a real attention-saturation effect, both datasets show the same
WER-vs-chunk-size curve shape. If chunking is just helping the model
retrieve memorized text, only everyayah shows the dramatic drop.

Tashkeel-stripped WER on every row (covers full Arabic diacritic block
plus Quranic small annotations).
"""
from __future__ import annotations
import argparse, gc, json, os, re, time, unicodedata
from typing import Optional
import numpy as np, psutil, soundfile as sf, jiwer

PER_RECITER = 50; MAX_RECITERS = 3; SR = 16000
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

def trim_leading_silence(samples, sr, energy_thresh=0.005, frame_ms=20):
    frame = int(frame_ms * sr / 1000)
    for i in range(0, len(samples), frame):
        chunk = samples[i:i+frame]
        if len(chunk) == 0: break
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        if rms > energy_thresh:
            return samples[i:]
    return samples

print('--- load fp32 NeMo ---', flush=True)
import sherpa_onnx
recog = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder='/tmp/rnnt/encoder.onnx',
    decoder='/tmp/rnnt/decoder.onnx',
    joiner='/tmp/rnnt/joiner.onnx',
    tokens='/tmp/rnnt/tokens.txt',
    num_threads=4, decoding_method='greedy_search',
    model_type='nemo_transducer')
print('NeMo loaded', flush=True)

vad_cfg = sherpa_onnx.VadModelConfig()
vad_cfg.silero_vad.model = '/tmp/rnnt/silero_vad.onnx'
vad_cfg.silero_vad.threshold = 0.5
vad_cfg.silero_vad.min_silence_duration = 0.3
vad_cfg.silero_vad.min_speech_duration = 0.2
vad_cfg.sample_rate = 16000
vad = sherpa_onnx.VoiceActivityDetector(config=vad_cfg, buffer_size_in_seconds=60)
print('VAD loaded', flush=True)

def run_decode(chunk):
    s = recog.create_stream()
    t0 = time.perf_counter()
    s.accept_waveform(SR, chunk.copy())
    recog.decode_stream(s)
    dt = time.perf_counter() - t0
    return norm(s.result.text), dt

def chunks_fixed(samples, win_ms, ov_ms, do_trim):
    win = int(win_ms * SR / 1000); ov = int(ov_ms * SR / 1000)
    step = max(1, win - ov)
    out = []
    for start in range(0, len(samples), step):
        c = samples[start:start+win]
        if len(c) < SR * 0.1: continue
        if do_trim:
            c = trim_leading_silence(c, SR)
            if len(c) < SR * 0.1: continue
        out.append(c)
        if start + win >= len(samples): break
    return out

def chunks_vad(samples, max_chunk_ms=None, do_trim=False, pad_ms=0):
    vad.reset()
    block = int(0.5 * SR)
    raw = []
    for i in range(0, len(samples), block):
        chunk = samples[i:i+block].copy()
        vad.accept_waveform(chunk)
        while not vad.empty():
            try:
                seg = vad.front if not callable(vad.front) else vad.front()
                ss = seg.samples if not callable(getattr(seg, 'samples', None)) else seg.samples()
                raw.append(np.asarray(ss, dtype='float32'))
            except Exception as e:
                print(f'  vad front err: {e}', flush=True)
            vad.pop()
    vad.flush()
    while not vad.empty():
        try:
            seg = vad.front if not callable(vad.front) else vad.front()
            ss = seg.samples if not callable(getattr(seg, 'samples', None)) else seg.samples()
            raw.append(np.asarray(ss, dtype='float32'))
        except Exception:
            pass
        vad.pop()
    out = []
    pad_samples = int(pad_ms * SR / 1000)
    for seg in raw:
        if max_chunk_ms is not None and len(seg) > int(max_chunk_ms * SR / 1000):
            mx = int(max_chunk_ms * SR / 1000)
            for i in range(0, len(seg), mx):
                sub = seg[i:i+mx]
                if len(sub) < SR * 0.1: continue
                if do_trim:
                    sub = trim_leading_silence(sub, SR)
                    if len(sub) < SR * 0.1: continue
                if pad_samples > 0:
                    sub = np.concatenate([np.zeros(pad_samples, dtype='float32'), sub])
                out.append(sub)
        else:
            if len(seg) < SR * 0.1: continue
            if do_trim:
                seg = trim_leading_silence(seg, SR)
                if len(seg) < SR * 0.1: continue
            if pad_samples > 0:
                seg = np.concatenate([np.zeros(pad_samples, dtype='float32'), seg])
            out.append(seg)
    return out

def transcribe(clips, chunker_fn, dedup=True):
    refs, hyps = [], []
    audio_s = decode_s = 0.0
    nrw = 0; per = {}; basmala = 0
    for c in clips:
        chunks = chunker_fn(c['samples'])
        all_words, last = [], []
        for chunk in chunks:
            text, dt = run_decode(chunk)
            decode_s += dt
            if not text: continue
            words = [w for w in text.split() if w]
            if dedup: words = boundary_dedup(last, words)
            all_words.extend(words)
            last = words
        hyp = ' '.join(all_words)
        audio_s += len(c['samples']) / SR
        refs.append(c['ref']); hyps.append(hyp)
        nrw += len(c['ref'].split())
        per.setdefault(c['subgroup'], []).append((c['ref'], hyp))
        if hyp.startswith('بسم') and not c['ref'].startswith('بسم'):
            basmala += 1
    return {
        'overall_wer': jiwer.wer(refs, hyps),
        'audio_seconds': audio_s,
        'decode_seconds': decode_s,
        'rtf': decode_s/audio_s if audio_s>0 else None,
        'words_per_sec': nrw/decode_s if decode_s>0 else None,
        'per_subgroup_wer': {r: jiwer.wer([x[0] for x in v], [x[1] for x in v]) for r,v in per.items() if v},
        'basmala_hallucinations': basmala,
        'n_clips': len(refs),
    }

# Datasets
print('--- cache everyayah ---', flush=True)
from datasets import load_dataset
ds = load_dataset('tarteel-ai/everyayah', split='train', streaming=True, token=os.environ['HF_TOKEN'])
everyayah = []; counts = {}
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
    everyayah.append({'samples': samples, 'sr': sr, 'subgroup': rec, 'ref': norm(row.get('text', ''))})
    if all(c >= PER_RECITER for c in counts.values()) and len(counts) >= MAX_RECITERS: break
print(f'everyayah: {len(everyayah)} clips', flush=True)

print('--- cache common_voice_17 (held-out) ---', flush=True)
cv = []
try:
    ds = load_dataset('mozilla-foundation/common_voice_17_0', 'ar', split='test',
                      streaming=True, token=os.environ['HF_TOKEN'], trust_remote_code=False)
    for row in ds:
        if len(cv) >= 150: break
        audio = row.get('audio')
        sentence = row.get('sentence', '')
        if audio is None or not sentence: continue
        sa = audio.get_all_samples()
        arr = sa.data.numpy()
        if arr.ndim > 1: arr = arr.mean(axis=0)
        arr = np.asarray(arr, dtype='float32')
        samples, sr = to16k(arr, sa.sample_rate)
        cv.append({'samples': samples, 'sr': sr, 'subgroup': 'cv-ar', 'ref': norm(sentence)})
    print(f'common_voice: {len(cv)} clips', flush=True)
except Exception as e:
    print(f'cv cache failed: {e}', flush=True)

datasets = {}
if cv: datasets['common_voice_17'] = cv
if everyayah:
    datasets['everyayah'] = everyayah

# Strategy grid — fine-grained around the 10s sweet spot
strategies = []
# QUICK HYPOTHESIS TEST: 10s/500ms + full-audio first — if chunking helps held-out,
# the effect is real attention saturation, not just data overlap.
strategies.append(("HYPOTHESIS_10000_500", lambda s: chunks_fixed(s, 10000, 500, False)))
strategies.append(("HYPOTHESIS_full",        lambda s: [s]))

# Fine grid around 8-12s
for win in [7000, 8000, 9000, 9500, 10000, 10500, 11000, 12000, 13000]:
    for ov in [0, 100, 200, 300, 500, 700, 1000]:
        if ov >= win: continue
        strategies.append((f'fixed_{win}_{ov}', lambda s, w=win, o=ov: chunks_fixed(s, w, o, False)))
# Wider grid for context
for win in [4000, 6000, 15000, 20000, 30000]:
    for ov in [0, 500]:
        if ov >= win: continue
        strategies.append((f'fixed_{win}_{ov}', lambda s, w=win, o=ov: chunks_fixed(s, w, o, False)))
# Silence-trim variants of the best fixed
for win in [9000, 10000, 11000]:
    for ov in [0, 500]:
        strategies.append((f'fixed_{win}_{ov}_trim', lambda s, w=win, o=ov: chunks_fixed(s, w, o, True)))
# Full audio
strategies.append(('full', lambda s: [s]))
# VAD variants
strategies.append(('vad_only', lambda s: chunks_vad(s, max_chunk_ms=None, do_trim=False)))
strategies.append(('vad_only_trim', lambda s: chunks_vad(s, max_chunk_ms=None, do_trim=True)))
for cap in [8000, 10000, 12000]:
    strategies.append((f'vad_cap{cap}', lambda s, c=cap: chunks_vad(s, max_chunk_ms=c, do_trim=False)))
    strategies.append((f'vad_cap{cap}_trim', lambda s, c=cap: chunks_vad(s, max_chunk_ms=c, do_trim=True)))
    strategies.append((f'vad_cap{cap}_pad200', lambda s, c=cap: chunks_vad(s, max_chunk_ms=c, do_trim=False, pad_ms=200)))

print(f'\n{len(strategies)} strategies × {len(datasets)} datasets = {len(strategies)*len(datasets)} configs', flush=True)
out_f = open('/tmp/results/kitchensink_v2.jsonl', 'a')
total = 0
for ds_name, clips in datasets.items():
    if not clips: continue
    for i, (name, fn) in enumerate(strategies):
        total += 1
        print(f'\n[{total}] {name} on {ds_name}', flush=True)
        gc.collect()
        try:
            res = transcribe(clips, fn)
            res['strategy'] = name
            res['dataset'] = ds_name
            res['model'] = 'nemo'
            out_f.write(json.dumps(res, ensure_ascii=False) + '\n')
            out_f.flush()
            print(f'  WER={res["overall_wer"]:.4f} RTF={res["rtf"]:.3f} wps={res["words_per_sec"]:.1f} بسم={res["basmala_hallucinations"]}', flush=True)
        except Exception as e:
            print(f'  err: {e}', flush=True)
out_f.close()
print('\n=== kitchensink v2 DONE ===', flush=True)

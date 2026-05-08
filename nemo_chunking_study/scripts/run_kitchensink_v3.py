"""Kitchen-sink v3 — adds ChunkFormer-inspired and WhisperX-style strategies.

New strategies vs v2:
  right_ctx_W_R       chunk size W ms with R ms of FUTURE audio appended; drop
                       the words covering the last R ms after decode (lookahead
                       trick — encoder gets future context, decoder output
                       trimmed by time)
  left_ctx_W_L        chunk size W ms with L ms of PAST audio prepended; drop
                       the words covering the first L ms after decode
  bidir_W_L_R         both — pad with L ms past + R ms future, drop both ends
  vad_merge_T         WhisperX-style: start from VAD segments, merge adjacent
                       shorts until reaching target length T ms
  vad_merge_T_ctx_R   merge to T ms then append R ms of right-context audio
"""
from __future__ import annotations
import gc, json, os, re, time, unicodedata
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

def words_in_time_range(words, total_ms, start_ms, end_ms):
    """Approximate which words fall within [start_ms, end_ms] of decoded text.
    Uses uniform time distribution (no per-word timestamps available)."""
    if not words: return words
    n = len(words)
    per_word = total_ms / n
    out = []
    for i, w in enumerate(words):
        word_start = i * per_word
        word_end = (i+1) * per_word
        # Keep word if it overlaps the kept range
        if word_end > start_ms and word_start < end_ms:
            out.append(w)
    return out

def chunks_with_ctx(samples, chunk_ms, left_ms, right_ms):
    """Build chunks with explicit left/right context. Returns list of
    (chunk_audio, drop_head_ms, drop_tail_ms) — the head/tail to remove
    from the decoded text after."""
    chunk_samp = int(chunk_ms * SR / 1000)
    left_samp = int(left_ms * SR / 1000)
    right_samp = int(right_ms * SR / 1000)
    out = []
    n = len(samples)
    for start in range(0, n, chunk_samp):
        end = min(start + chunk_samp, n)
        # Effective audio range with context
        eff_start = max(0, start - left_samp)
        eff_end = min(n, end + right_samp)
        chunk_audio = samples[eff_start:eff_end]
        if len(chunk_audio) < SR * 0.1: continue
        # How many ms of decoded text correspond to left/right context
        drop_head_ms = (start - eff_start) * 1000 // SR
        drop_tail_ms = (eff_end - end) * 1000 // SR
        out.append((chunk_audio, drop_head_ms, drop_tail_ms))
        if end >= n: break
    return out

def vad_merge_chunks(samples, target_ms, right_ctx_ms=0):
    """WhisperX-style: get VAD segments, merge consecutive ones until each
    reaches at least target_ms. Optionally append right_ctx_ms of audio
    AFTER the merged segment for encoder context."""
    vad.reset()
    block = int(0.5 * SR)
    raw = []
    for i in range(0, len(samples), block):
        chunk = samples[i:i+block].copy()
        vad.accept_waveform(chunk)
        while not vad.empty():
            seg = vad.front if not callable(vad.front) else vad.front()
            try:
                ss = seg.samples if not callable(getattr(seg, 'samples', None)) else seg.samples()
                start = seg.start if hasattr(seg, 'start') else 0
                raw.append((np.asarray(ss, dtype='float32'), start))
            except Exception:
                pass
            vad.pop()
    vad.flush()
    while not vad.empty():
        seg = vad.front if not callable(vad.front) else vad.front()
        try:
            ss = seg.samples if not callable(getattr(seg, 'samples', None)) else seg.samples()
            start = seg.start if hasattr(seg, 'start') else 0
            raw.append((np.asarray(ss, dtype='float32'), start))
        except Exception:
            pass
        vad.pop()
    if not raw: return []
    # Merge
    target_samp = int(target_ms * SR / 1000)
    right_samp = int(right_ctx_ms * SR / 1000)
    merged = []
    cur_audio = []
    cur_end = -1
    for seg_audio, seg_start in raw:
        if cur_end >= 0 and len(np.concatenate(cur_audio)) < target_samp:
            cur_audio.append(seg_audio)
            cur_end = seg_start + len(seg_audio)
        else:
            if cur_audio:
                merged.append(np.concatenate(cur_audio))
            cur_audio = [seg_audio]
            cur_end = seg_start + len(seg_audio)
    if cur_audio:
        merged.append(np.concatenate(cur_audio))
    # Append right context if requested — pad with silence for now since we
    # don't have access to the original sample positions reliably
    if right_samp > 0:
        merged = [np.concatenate([m, np.zeros(right_samp, dtype='float32')]) for m in merged]
    return merged

def transcribe_with_drop(clips, chunker_fn):
    """For chunkers that return (audio, drop_head_ms, drop_tail_ms) tuples."""
    refs, hyps = [], []
    audio_s = decode_s = 0.0; nrw = 0; per = {}; basmala = 0
    for c in clips:
        chunks_data = chunker_fn(c['samples'])
        all_words = []
        for chunk_audio, drop_head, drop_tail in chunks_data:
            text, dt = run_decode(chunk_audio)
            decode_s += dt
            if not text: continue
            words = [w for w in text.split() if w]
            chunk_total_ms = len(chunk_audio) * 1000 // SR
            kept = words_in_time_range(words, chunk_total_ms,
                                         drop_head, chunk_total_ms - drop_tail)
            all_words.extend(kept)
        hyp = ' '.join(all_words)
        audio_s += len(c['samples']) / SR
        refs.append(c['ref']); hyps.append(hyp)
        nrw += len(c['ref'].split())
        per.setdefault(c['subgroup'], []).append((c['ref'], hyp))
        if hyp.startswith('بسم') and not c['ref'].startswith('بسم'): basmala += 1
    return {
        'overall_wer': jiwer.wer(refs, hyps),
        'audio_seconds': audio_s, 'decode_seconds': decode_s,
        'rtf': decode_s/audio_s if audio_s>0 else None,
        'words_per_sec': nrw/decode_s if decode_s>0 else None,
        'per_subgroup_wer': {r: jiwer.wer([x[0] for x in v], [x[1] for x in v]) for r,v in per.items() if v},
        'basmala_hallucinations': basmala,
        'n_clips': len(refs),
    }

def transcribe_simple(clips, chunker_fn):
    """For chunkers that return plain audio chunks (no drop info)."""
    refs, hyps = [], []
    audio_s = decode_s = 0.0; nrw = 0; per = {}; basmala = 0
    for c in clips:
        chunks_audio = chunker_fn(c['samples'])
        all_words = []
        for chunk in chunks_audio:
            text, dt = run_decode(chunk)
            decode_s += dt
            if not text: continue
            words = [w for w in text.split() if w]
            all_words.extend(words)
        hyp = ' '.join(all_words)
        audio_s += len(c['samples']) / SR
        refs.append(c['ref']); hyps.append(hyp)
        nrw += len(c['ref'].split())
        per.setdefault(c['subgroup'], []).append((c['ref'], hyp))
        if hyp.startswith('بسم') and not c['ref'].startswith('بسم'): basmala += 1
    return {
        'overall_wer': jiwer.wer(refs, hyps),
        'audio_seconds': audio_s, 'decode_seconds': decode_s,
        'rtf': decode_s/audio_s if audio_s>0 else None,
        'words_per_sec': nrw/decode_s if decode_s>0 else None,
        'per_subgroup_wer': {r: jiwer.wer([x[0] for x in v], [x[1] for x in v]) for r,v in per.items() if v},
        'basmala_hallucinations': basmala,
        'n_clips': len(refs),
    }

# Cache datasets
print('--- cache 150 clips ---', flush=True)
from datasets import load_dataset
ds = load_dataset('tarteel-ai/everyayah', split='train', streaming=True, token=os.environ['HF_TOKEN'])
clips = []; counts = {}
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
    clips.append({'samples': samples, 'sr': sr, 'subgroup': rec, 'ref': norm(row.get('text', ''))})
    if all(c >= PER_RECITER for c in counts.values()) and len(counts) >= MAX_RECITERS: break
print(f'cached {len(clips)} clips', flush=True)

# Strategy grid — focused on the new techniques
strategies_with_drop = []  # (name, fn) — fn returns list of (audio, drop_head_ms, drop_tail_ms)
strategies_simple = []     # (name, fn) — fn returns list of plain audio chunks

# Right-context only: chunk W ms + right_ms of future audio
for win in [6000, 8000, 10000]:
    for right in [500, 1000, 2000, 3000]:
        strategies_with_drop.append(
            (f'right_ctx_{win}_{right}',
             lambda s, w=win, r=right: chunks_with_ctx(s, w, 0, r)))
# Left-context only
for win in [8000, 10000]:
    for left in [500, 1000, 2000]:
        strategies_with_drop.append(
            (f'left_ctx_{win}_{left}',
             lambda s, w=win, l=left: chunks_with_ctx(s, w, l, 0)))
# Bidirectional context
for win in [6000, 8000, 10000]:
    for both in [(500,500), (1000,1000), (1000,2000), (2000,1000)]:
        l, r = both
        strategies_with_drop.append(
            (f'bidir_{win}_{l}_{r}',
             lambda s, w=win, l=l, r=r: chunks_with_ctx(s, w, l, r)))

# WhisperX-style VAD merge
for target in [6000, 8000, 10000, 12000]:
    strategies_simple.append(
        (f'vad_merge_{target}',
         lambda s, t=target: vad_merge_chunks(s, t, 0)))
    for right in [500, 1000, 2000]:
        strategies_simple.append(
            (f'vad_merge_{target}_ctx_{right}',
             lambda s, t=target, r=right: vad_merge_chunks(s, t, r)))

print(f'\n{len(strategies_with_drop)} drop strategies + {len(strategies_simple)} simple = {len(strategies_with_drop)+len(strategies_simple)} total', flush=True)
out_f = open('/tmp/results/kitchensink_v3.jsonl', 'a')
total = 0
for name, fn in strategies_with_drop:
    total += 1
    print(f'\n[{total}] {name}', flush=True)
    gc.collect()
    try:
        res = transcribe_with_drop(clips, fn)
        res['strategy'] = name; res['model'] = 'nemo'; res['dataset'] = 'everyayah'
        out_f.write(json.dumps(res, ensure_ascii=False) + '\n'); out_f.flush()
        print(f'  WER={res["overall_wer"]:.4f} RTF={res["rtf"]:.3f} wps={res["words_per_sec"]:.1f} بسم={res["basmala_hallucinations"]}', flush=True)
    except Exception as e:
        print(f'  err: {e}', flush=True)
for name, fn in strategies_simple:
    total += 1
    print(f'\n[{total}] {name}', flush=True)
    gc.collect()
    try:
        res = transcribe_simple(clips, fn)
        res['strategy'] = name; res['model'] = 'nemo'; res['dataset'] = 'everyayah'
        out_f.write(json.dumps(res, ensure_ascii=False) + '\n'); out_f.flush()
        print(f'  WER={res["overall_wer"]:.4f} RTF={res["rtf"]:.3f} wps={res["words_per_sec"]:.1f} بسم={res["basmala_hallucinations"]}', flush=True)
    except Exception as e:
        print(f'  err: {e}', flush=True)
out_f.close()
print('\n=== v3 DONE ===', flush=True)

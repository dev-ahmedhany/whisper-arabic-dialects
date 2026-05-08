"""NeMo top-strategies on real-human held-out Arabic dataset.
Hardcoded to MGB-3 variants — both real Egyptian human speech, NOT in NeMo training."""
import os, re, json, time, unicodedata, argparse
import numpy as np, soundfile as sf, jiwer, sherpa_onnx
from datasets import load_dataset

PER_CLIP = 100; SR = 16000
TASHKEEL = re.compile(r'[ً-ٰۖ-ۭ]')

def norm(t):
    t = unicodedata.normalize('NFC', t or '')
    t = TASHKEEL.sub('', t)
    t = re.sub(r'[أإآٱ]', 'ا', t)
    t = t.replace('ى', 'ي').replace('ة', 'ه')
    return re.sub(r'\s+', ' ', t).strip()

def to16k(s, sr):
    if s.ndim > 1: s = s.mean(axis=1)
    if sr != SR:
        ratio = SR/sr
        idx = np.linspace(0, len(s)-1, int(len(s)*ratio)).astype('int')
        s = s[idx].astype('float32'); sr = SR
    return s, sr

def boundary_dedup(prev, new, max_n=5):
    if not prev or not new: return new
    n_max = min(max_n, len(prev), len(new))
    for n in range(n_max, 0, -1):
        if prev[-n:] == new[:n]: return new[n:]
    return new

ap = argparse.ArgumentParser()
ap.add_argument('--dataset-path', required=True)
ap.add_argument('--split', default='test')
ap.add_argument('--text-field', default='transcript')
ap.add_argument('--out', default='/tmp/results/holdout_mgb.jsonl')
args = ap.parse_args()

recog = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder='/tmp/rnnt/encoder.onnx', decoder='/tmp/rnnt/decoder.onnx',
    joiner='/tmp/rnnt/joiner.onnx', tokens='/tmp/rnnt/tokens.txt',
    num_threads=4, decoding_method='greedy_search', model_type='nemo_transducer')
print(f'NeMo loaded')

def transcribe(samples):
    s = recog.create_stream()
    t0 = time.perf_counter()
    s.accept_waveform(SR, samples.copy())
    recog.decode_stream(s)
    return norm(s.result.text), time.perf_counter()-t0

def chunks_fixed(samples, win_ms, ov_ms):
    win = int(win_ms*SR/1000); ov = int(ov_ms*SR/1000); step = max(1, win-ov)
    out = []
    for start in range(0, len(samples), step):
        c = samples[start:start+win]
        if len(c) < SR*0.1: continue
        out.append(c)
        if start + win >= len(samples): break
    return out

print(f'--- cache {args.dataset_path} split={args.split} text-field={args.text_field} ---')
ds = load_dataset(args.dataset_path, split=args.split, streaming=True,
                  token=os.environ.get('HF_TOKEN'), trust_remote_code=True)
clips = []
for row in ds:
    if len(clips) >= PER_CLIP: break
    audio = row.get('audio')
    text = row.get(args.text_field) or row.get('text') or row.get('transcription') or ''
    if not audio or not text: continue
    arr = np.asarray(audio['array'], dtype='float32')
    samples, sr = to16k(arr, audio['sampling_rate'])
    if len(samples) < SR * 0.5: continue  # skip <0.5s clips
    clips.append({'samples': samples, 'ref': norm(text)})
print(f'cached {len(clips)} clips')
if not clips: exit(1)

# Show a sanity sample
print(f'\nfirst clip ref:  {clips[0]["ref"][:80]}')
t, _ = transcribe(clips[0]['samples'])
print(f'first clip nemo: {t[:80]}\n')

strategies = [
    ('full',                lambda s: [s]),
    ('fixed_11000_100',     lambda s: chunks_fixed(s, 11000, 100)),
    ('fixed_10500_100',     lambda s: chunks_fixed(s, 10500, 100)),
    ('fixed_10000_500',     lambda s: chunks_fixed(s, 10000, 500)),
    ('fixed_8000_500',      lambda s: chunks_fixed(s, 8000, 500)),
    ('fixed_4000_0',        lambda s: chunks_fixed(s, 4000, 0)),
]

os.makedirs(os.path.dirname(args.out), exist_ok=True)
out_f = open(args.out, 'a')
for name, fn in strategies:
    refs, hyps, audio_s, decode_s = [], [], 0, 0
    for c in clips:
        ws, last = [], []
        for chunk in fn(c['samples']):
            t, dt = transcribe(chunk); decode_s += dt
            if not t: continue
            w = [x for x in t.split() if x]
            w = boundary_dedup(last, w); ws.extend(w); last = w
        hyp = ' '.join(ws)
        refs.append(c['ref']); hyps.append(hyp)
        audio_s += len(c['samples'])/SR
    wer = jiwer.wer(refs, hyps)
    res = {'dataset': args.dataset_path, 'strategy': name, 'wer': wer,
           'rtf': decode_s/audio_s, 'n_clips': len(refs)}
    out_f.write(json.dumps(res, ensure_ascii=False)+'\n'); out_f.flush()
    print(f'  {name:25s}  WER={wer*100:6.2f}%  RTF={res["rtf"]:.4f}')

print('=== DONE ===')

"""whisper-large-v3 chunking grid on SADA22 MSA held-out — same protocol as NeMo SADA22."""
from __future__ import annotations
import argparse, gc, json, os, re, time, unicodedata
import numpy as np, jiwer

PER_CLIP = 100; SR = 16000; MIN_DUR = 8.0
TASHKEEL = re.compile(r'[ً-ٰۖ-ۭ]')

def norm(t):
    t = unicodedata.normalize('NFC', t or '')
    t = TASHKEEL.sub('', t)
    t = re.sub(r'[أإآٱ]', 'ا', t)
    t = t.replace('ى', 'ي').replace('ة', 'ه')
    t = re.sub(r'[.,،;؛؟!?\-\(\)\[\]"\'`]', '', t)
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
ap.add_argument('--model-id', default='openai/whisper-large-v3')
ap.add_argument('--model-name', default='whisper-large-v3')
ap.add_argument('--out', default='/tmp/results/whisper_sada22.jsonl')
args = ap.parse_args()

print(f'--- load {args.model_name} ---', flush=True)
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32
print(f'device={DEVICE} dtype={DTYPE}', flush=True)

proc_obj = WhisperProcessor.from_pretrained(args.model_id)
model = WhisperForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=DTYPE).to(DEVICE)
model.eval()
if hasattr(model.generation_config, 'language'):
    model.generation_config.language = 'arabic'
if hasattr(model.generation_config, 'task'):
    model.generation_config.task = 'transcribe'
print(f'{args.model_name} loaded', flush=True)

def transcribe(samples):
    inputs = proc_obj(samples, sampling_rate=SR, return_tensors='pt')
    feats = inputs.input_features.to(DEVICE).to(DTYPE)
    if DEVICE == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        ids = model.generate(feats, num_beams=1, do_sample=False, max_new_tokens=256)
    if DEVICE == 'cuda': torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    text = proc_obj.batch_decode(ids, skip_special_tokens=True)[0]
    return norm(text), dt

def chunks_fixed(samples, win_ms, ov_ms):
    win = int(win_ms*SR/1000); ov = int(ov_ms*SR/1000); step = max(1, win-ov)
    out = []
    for start in range(0, len(samples), step):
        c = samples[start:start+win]
        if len(c) < SR*0.1: continue
        out.append(c)
        if start + win >= len(samples): break
    return out

print(f'--- cache SADA22 MSA min-dur={MIN_DUR}s ---', flush=True)
from datasets import load_dataset
ds = load_dataset('badrex/arabic-speech-SADA22-MSA', split='train', streaming=True,
                  token=os.environ.get('HF_TOKEN'), trust_remote_code=True)
clips = []; scanned = 0
for row in ds:
    scanned += 1
    if len(clips) >= PER_CLIP: break
    if scanned > 5000: break
    audio = row.get('audio')
    text = row.get('cleaned_text') or row.get('text') or ''
    if not audio or not text: continue
    arr = np.asarray(audio['array'], dtype='float32')
    samples, _ = to16k(arr, audio['sampling_rate'])
    dur = len(samples) / SR
    if dur < MIN_DUR: continue
    clips.append({'samples': samples, 'ref': norm(text), 'dur': dur})
print(f'cached {len(clips)} clips (scanned {scanned})', flush=True)

# Sanity check
samples = clips[0]['samples']
text, _ = transcribe(samples)
print(f'\nfirst ref:  {clips[0]["ref"][:90]}')
print(f'first nemo: {text[:90]}\n', flush=True)

strategies = [
    ('full',                lambda s: [s]),
    ('fixed_30000_500',     lambda s: chunks_fixed(s, 30000, 500)),
    ('fixed_20000_500',     lambda s: chunks_fixed(s, 20000, 500)),
    ('fixed_15000_500',     lambda s: chunks_fixed(s, 15000, 500)),
    ('fixed_11000_100',     lambda s: chunks_fixed(s, 11000, 100)),
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
    res = {'dataset': 'sada22-msa-min8s', 'model': args.model_name,
           'strategy': name, 'wer': wer, 'rtf': decode_s/audio_s, 'n_clips': len(refs)}
    out_f.write(json.dumps(res, ensure_ascii=False)+'\n'); out_f.flush()
    print(f'  {name:25s}  WER={wer*100:6.2f}%  RTF={res["rtf"]:.4f}', flush=True)

print('=== DONE ===', flush=True)

"""Prepare QASR (Aljazeera) → JSONL. Filtered by speaker dialect column where available.

QASR is hosted by QCRI; access via http://www.openslr.org/106/ (free, no registration)
or via QCRI-Hamad MGB-2/QASR distribution (license terms apply). Pass the local
unzipped directory.

Expected layout (after unzip):
  <root>/audio/{split}/<utt-id>.wav
  <root>/text/{split}/text                 (Kaldi-style)
  <root>/text/{split}/utt2spk              (utt -> speaker)
  <root>/spk2dialect (optional)             (speaker -> dialect)

If spk2dialect is absent, --default-dialect is applied to all rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from scripts.prepare_mgb3 import parse_kaldi_text


def _load_two_col(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="QASR unzipped root")
    p.add_argument("--split", default="train")
    p.add_argument("--default-dialect", default="levantine")
    p.add_argument("--filter-dialect", default=None,
                   help="if set, only include rows whose speaker maps to this dialect")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-tag", default="qasr")
    args = p.parse_args()

    audio_dir = args.root / "audio" / args.split
    text_file = args.root / "text" / args.split / "text"
    utt2spk = _load_two_col(args.root / "text" / args.split / "utt2spk")
    spk2dialect = _load_two_col(args.root / "spk2dialect")
    transcripts = parse_kaldi_text(text_file)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w") as out:
        for utt_id, ref in transcripts.items():
            wav = audio_dir / f"{utt_id}.wav"
            if not wav.exists():
                continue
            spk = utt2spk.get(utt_id)
            dialect = spk2dialect.get(spk, args.default_dialect) if spk else args.default_dialect
            if args.filter_dialect and dialect != args.filter_dialect:
                continue
            info = sf.info(str(wav))
            duration_s = float(info.frames) / float(info.samplerate)
            out.write(
                json.dumps(
                    {
                        "audio": str(wav.resolve()),
                        "reference": ref.strip(),
                        "dialect": dialect,
                        "source_dataset": args.source_tag,
                        "duration_s": duration_s,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()

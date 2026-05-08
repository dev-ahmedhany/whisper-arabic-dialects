# LinkedIn post — Arabic, viral-style (template, fill in once results land)

> Placeholders in `{ }` get replaced by the headline numbers from the bench.
> Goal: a post that engineers reshare ("we needed this") and that
> non-engineers can still understand the headline of.

---

🚨 وفّرنا **{NEMO_CHUNK_DELTA_PP}** نقطة WER على نموذج عربي
مفتوح المصدر، **بدون أي fine-tuning** — مجرد طريقة تقطيع ذكية للصوت.

والمفاجأة الحقيقية: **لا تنفع هذه الحيلة مع Whisper.**

---

السياق: نبني تطبيق على الجوال يستمع للقارئ ويكتشف أخطاء التلاوة
لحظيًا. النموذج اللي اخترناه هو
`nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` — 115M معلمة،
على CPU، RNNT greedy، صفر LM bias (مهم لكشف الأخطاء بدل تصحيحها
تلقائيًا).

من الورقة (NVIDIA) WER ~7% على MSA. لكن على القرآن: **27.25% WER
على ملف كامل**، يقفز لـ **41.8% على القراءة المجوّدة**.

---

🤔 لماذا؟ ليس لأن النموذج "سيء". بل لأن الـ encoder تدرّب على مقاطع
≤ 10–15 ثانية. القراءة المجوّدة تتجاوز ذلك (آيات تصل لـ 25+ ثانية بدون
وقفة)، فتشبع الـ attention والـ positional encoding ويبدأ الإخراج
بالتدهور.

**التشخيص السريع:** هل التقطيع لمقاطع 10 ثانية يحسن أم يسوء النتيجة؟

---

📊 نتائج 150 آية × 3 قرّاء × 9 أحجام تقطيع × نموذجين (e2-standard-16
على GCP):

| الإعداد | WER |
|---|---:|
| FastConformer-AR ملف كامل | 27.25% |
| **+ تقطيع 10 ث × 500 ms overlap** | **{NEMO_BEST_WER}%** ⭐ |
| {WHISPER_NAME} ملف كامل | {WHISPER_FULL_WER}% |
| {WHISPER_NAME} + نفس التقطيع | {WHISPER_CHUNKED_WER}% |

📌 **النتيجة الكبرى:** نفس التقطيع يفيد FastConformer (-{NEMO_DELTA} pp)
ولا يفيد Whisper (Δ {WHISPER_DELTA} pp).

---

💡 **لماذا الفرق؟**

- **FastConformer** encoder + RNNT decoder: encoder تدرّب على مقاطع
  قصيرة، الـ chunking يعيده للنطاق المريح + يمسح الـ "Bismillah
  hallucination" (وهي أن الـ RNNT يبدأ من `<s>` ويتنبأ بـ "بسم الله"
  حتى لو القارئ بدأ من نص الآية).
- **Whisper** encoder-decoder: تدرّب أصلاً على 30 ثانية، فلا
  يستفيد من التقطيع.

🧩 الدرس المعمم: **التقطيع ليس "ترك الأمر للمكتبة"، بل قرار معماري.**
Whisper-streaming popularized LocalAgreement-2، لكن الخوارزمية تفترض
نمطًا مختلفًا للنوافذ (growing-window). إذا استخدمتها مع نوافذ ثابتة
(fixed-sliding window) فلن تثبّت أي كلمة. **تجربتنا: 96.91% WER
بنفس النموذج لمجرد الخوارزمية الخاطئة.**

استبدلناها بخوارزمية أبسط (`boundary dedup`: نسقط الـ longest n-gram
الذي يكرّره فجوة الـ overlap)، النتيجة هبطت من 96.91% إلى **17.33%**
في نفس الإعداد.

---

🛠 **الـ stack الكامل (مفتوح المصدر):**

- ONNX export: `sherpa-onnx/scripts/nemo/fast-conformer-hybrid-transducer-ctc/`
- Mobile inference: `sherpa-onnx` Flutter plugin
- Chunking + dedup: `nemo_streaming` Dart package (extراج من تطبيق
  Murattil — رابط الـ repo في التعليقات)
- Eval harness: نفس الإطار اللي بنينا للـ Whisper-Arabic-Dialects
  paper (Tashkeel-stripped WER, hardware fingerprint, JSONL logging)

📂 الكود + الـ JSONL + التحليل الكامل:
github.com/dev-ahmedhany/whisper-arabic-dialects/tree/main/nemo_chunking_study

---

🤲 من فعل خير: لو الورقة تخدمك في تطبيق ASR للقرآن أو غيره، شارك
خبرتك في التعليقات. الـ insight اللي أذهلني أكثر هو {INSIGHT}.

#ArabicNLP #ASR #SpeechRecognition #OnDeviceAI #NeMo #Whisper
#OpenSource #Quran #الذكاء_الاصطناعي

---

## Notes (NOT for publishing)

- Replace `{NEMO_CHUNK_DELTA_PP}` with the live-default Δ vs full-audio
  (e.g. "9.92" for 27.25 → 17.33).
- `{NEMO_BEST_WER}` = best WER for NeMo on everyayah (likely 12.64% at
  10s/500ms or lower from the wider sweep).
- `{WHISPER_NAME}` = the most-relevant Whisper variant from the bench
  (probably `whisper-base-ar-quran` since it's both the fairest baseline
  AND the comparison point).
- `{WHISPER_DELTA}` = signed delta (probably +0.X pp = chunking hurt
  Whisper, validating the "different architectures, different needs"
  framing).
- `{INSIGHT}` = whichever finding lands strongest:
  - "أن نفس النموذج يقفز من 41% إلى ~13% على القراءة المجوّدة بمجرد التقطيع"
  - "أن LocalAgreement-2 خوارزمية صحيحة لـ pattern معيّن وكارثية لآخر"
  - "أن Whisper لا يستفيد من التقطيع لأن سياقه التدريبي أصلاً 30 ثانية"
- The post is intentionally heavy on numbers + light on jargon; the
  goal is "engineer says: I'm forwarding this to my team" and
  "non-engineer says: oh, that's a clever trick I didn't know".


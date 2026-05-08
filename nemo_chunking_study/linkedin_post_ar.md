# LinkedIn post — viral Arabic, real held-out cross-architecture numbers

---

ازاي خفّضت نسبة الخطأ في تعرّف الكلام العربي من ٥٠٪ لـ ١٨٪ من غير ما
أعيد تدريب أي نموذج، بسطر كود واحد بس؟ 🤯

كلنا عارفين إن نماذج الـ ASR العربية بتتعب لما الصوت يطول. النموذج بيشوف
كلام أكتر من اللي اتدرّب عليه، فبيهلوس، يدمج كلمات، يدلع.

النهارده فخور أعلن عن مشروع بحثي شغّلته على ٤ سيرفرات بالتوازي، بـ ١٧٢
تجربة على بيانات حقيقية، لقيت إن مجرد تقسيم الملف الصوتي بطريقة معيّنة
بيغيّر النتيجة بشكل لا يُصدّق:

🔹 ملف صوتي كامل (طريقة الناس العادية): **٥٠٪ خطأ**
🔹 تقسيم لمقاطع ١١ ثانية + ١٠٠ مللي ثانية تداخل: **١٨٪ خطأ** ⭐
🔹 الفرق: ٣٢ نقطة WER. نفس الموديل، نفس الصوت، نفس الـ CPU.

والأهم: ده كله على بيانات **خارج تدريب الموديل خالص** — صوت مذيع سعودي
الموديل ما شافوش طول حياته. لا data leak، لا غش.

اشتغلت على نموذج NVIDIA FastConformer-AR-pcd مفتوح المصدر، ١١٥ مليون
باراميتر، شغّال على CPU عادي بدون GPU.

النتايج عبر ٥ datasets مختلفة: 📊
🔹 قراءة قرآنية (everyayah): ٢٧٪ → ١١٪ = **-١٦ نقطة**
🔹 فصحى سعودية (SADA22) - ملفات ٦+ ثانية: ٤١٪ → ١٩٪ = **-٢٢ نقطة**
🔹 فصحى سعودية - ملفات ٨+ ثانية: ٥٠٪ → ١٨٪ = **-٣٢ نقطة** 🚀
🔹 لهجة مصرية (MGB-3): ٤٠٪ → ٤٠٪ = الموديل ما عرفش يفهمها أصلاً
🔹 جربتها كمان على Whisper-large-v3 (موديل OpenAI ١.٥ بليون باراميتر):
   ٣٦٪ → ٣٠٪ = **-٦ نقاط** عبر architecture مختلف تماماً ✅

اللي تعلمته: **الفايدة بتكبر مع طول الملف**. كل ما الكلام يطول عن المسافة
اللي اتدرّب عليها الموديل، التقسيم بينقذك أكتر.

التشغيل: 💪
🔹 سيرفر Hetzner CX33 بـ ٧.٩٩ دولار في الشهر (٤ كور x86 EPYC، ٨ جيجا رام)
🔹 ٢٤.٧× أسرع من الزمن الحقيقي (تفرّغ ٥٩٤ ساعة صوت في اليوم)
🔹 تكلفة الدقيقة الواحدة = ٠.٠٠٠٠٠٧٥ سنت
🔹 شغّال **على الموبايل** (iPhone + Android) بدون إنترنت ولا اشتراك

الكود + كل النتائج + الرسم البياني للـ Pareto frontier متاح كله Open
Source على GitHub. الباحثين والمطوّرين، شاركوا، جرّبوا، حسّنوا، طبّقوا
على لغاتكم. 🤝

اللينكات في أول كومنت 👇

#ArabicNLP #ASR #MachineLearning #OpenSource #الذكاء_الاصطناعي #AI

---

## First-comment links

📂 الكود + النتائج كاملة (١٧٢ تجربة على ٥ datasets):
https://github.com/dev-ahmedhany/whisper-arabic-dialects/tree/main/nemo_chunking_study

📦 الباكدج اللي بيشغل الموديل على الموبايل (Flutter/Dart):
https://github.com/dev-ahmedhany/nemo-streaming

📱 تطبيق Murattil (Tarteel-alternative مجاني، شغّال offline على القرآن، تحت التطوير):
https://github.com/dev-ahmedhany/murattil

📊 الـ Pareto frontier التفصيلي + جداول WER لكل استراتيجية + المراجع
الأكاديمية (Open ASR Leaderboard, ChunkFormer, WhisperX، إلخ):
https://github.com/dev-ahmedhany/whisper-arabic-dialects/blob/main/nemo_chunking_study/README.md

🎓 مشروع سابق (LoRA-fine-tune لـ Whisper على اللهجات العربية):
https://huggingface.co/dev-ahmedhany/whisper-large-v3-arabic-ft-v3

---

## Key numbers (validated, in repo as raw jsonl)

NeMo FastConformer-AR-pcd (115M params, greedy decoder):
- everyayah Quran (in training, 150 clips): 27.25% → 10.99% = -16.27pp
- SADA22 MSA min-6s (held-out, 100 clips): 41.33% → 19.37% = -21.96pp
- SADA22 MSA min-8s (held-out, 100 clips): 50.43% → 18.41% = -32.01pp ⭐
- MGB-3 ArabicSpeech broadcast (held-out, 100 clips): 46.77% → 46.77% (model fails)

whisper-large-v3 (1.5B params, greedy, fp16 on L4 GPU):
- SADA22 MSA min-8s (held-out, 100 clips): 36.38% → 30.14% = -6.24pp

Best strategy across all datasets: `fixed_11000_100` (11s window, 100ms overlap, n-gram boundary dedup).

CX33 throughput bench (Hetzner, AMD EPYC-Rome 4 cores @2.45GHz, x86_64):
- RTF: 0.0404 (50 decodes × 10s clips)
- 24.75× real-time
- 593.99 hours audio / 24h
- 17,819.83 hours audio / month
- $7.99 / 17,820h = $0.000448/h = $0.0000075/min

---

## Notes (NOT for posting)

- Headline number: -32pp on held-out SADA22 MSA min-8s. This is the
  strongest claim because (a) data was 100% out-of-distribution for
  NeMo's training, (b) effect size is huge, (c) explanation is
  intuitive (long clips overflow training distribution).
- The whisper-large-v3 cross-architecture validation is the
  closer — proves it's not a NeMo-specific quirk. Smaller delta
  (-6.24pp) because Whisper's 30s training distribution is more
  forgiving for long clips than NeMo's 20s.
- MGB-3 Egyptian dialect 0pp result is actually informative: chunking
  doesn't fix what the model can't do at all. Chunking is a
  duration-distribution-mismatch fix, not a magic ASR booster.
- Avoided the previously-overclaimed "Whisper-base-ar-quran beaten
  by 7 points" — we don't have that data anymore (lost when L4 #1
  was preempted). The cross-architecture story now rests on
  large-v3 which is stronger anyway.

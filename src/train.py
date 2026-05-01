"""QLoRA fine-tuning of Whisper variants on dialect-balanced Arabic.

Recipe: NF4 4-bit base + r=32 / alpha=64 LoRA on q/k/v/out_proj + fc1/fc2, bf16 compute,
paged_adamw_8bit, lr=1e-4, 3 epochs, flash-attention 2 on L4. Encoder is NOT frozen —
needed for accent adaptation.

Driven by a YAML config; CLI flags can override any top-level key.

Usage:
    python -m src.train --config configs/train_turbo.yaml
    python -m src.train --config configs/train_turbo.yaml --num-train-epochs 1 --max-steps 200  # sanity
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Audio, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

import jiwer
from src.normalization import NORMALIZER_VERSION, normalize_arabic


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Pad mel features. feature_extractor.pad returns a BatchFeature dict;
        # in some transformers versions it leaks an `input_ids` key that Whisper's
        # forward() rejects. Explicitly extract only the field we want.
        input_features = [{"input_features": f["input_features"]} for f in features]
        feat_batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels.
        labels_list = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(labels_list, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # Return ONLY what Whisper.forward() accepts — no input_ids leak.
        return {
            "input_features": feat_batch["input_features"],
            "labels": labels,
        }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git"


def _load_config(config_path: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text())
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def _prepare_dataset(processor, train_jsonl: str, val_jsonl: str, audio_column: str = "audio"):
    ds = load_dataset(
        "json",
        data_files={"train": train_jsonl, "validation": val_jsonl},
    )
    ds = ds.cast_column(audio_column, Audio(sampling_rate=16000))

    def _prepare(batch):
        audio = batch[audio_column]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["reference"]).input_ids
        return batch

    cols_to_remove = [c for c in ds["train"].column_names if c not in ("input_features", "labels")]
    ds = ds.map(_prepare, remove_columns=cols_to_remove, num_proc=1)
    return ds


def build_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = [normalize_arabic(s) for s in pred_str]
        label_str = [normalize_arabic(s) for s in label_str]
        return {"wer": jiwer.wer(label_str, pred_str)}

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(
        args.config,
        overrides={
            "num_train_epochs": args.num_train_epochs,
            "max_steps": args.max_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "learning_rate": args.learning_rate,
            "output_dir": args.output_dir,
        },
    )

    model_name = cfg["model_name"]
    processor = WhisperProcessor.from_pretrained(
        model_name, language=cfg.get("language", "arabic"), task=cfg.get("task", "transcribe")
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # PEFT + Seq2SeqTrainer leaks several kwargs into Whisper that Whisper
    # doesn't accept. We patch BOTH Whisper.forward AND Whisper.generate.
    #
    # forward leakage (PEFT's PeftModelForSeq2SeqLM.forward passes these):
    #   input_ids, inputs_embeds, task_ids
    #
    # generate leakage (Seq2SeqTrainer.prediction_step passes the inputs dict
    # that includes labels, which WhisperForConditionalGeneration.generate's
    # _validate_model_kwargs rejects):
    #   labels
    if not getattr(WhisperForConditionalGeneration.forward, "_peft_compat_patched", False):
        _orig_whisper_forward = WhisperForConditionalGeneration.forward
        _orig_whisper_generate = WhisperForConditionalGeneration.generate
        _FORWARD_LEAKED = ("input_ids", "inputs_embeds", "task_ids")
        _GENERATE_LEAKED = ("labels",)

        def _whisper_forward_peft_compat(self, *args, **kwargs):
            for k in _FORWARD_LEAKED:
                kwargs.pop(k, None)
            return _orig_whisper_forward(self, *args, **kwargs)

        def _whisper_generate_peft_compat(self, *args, **kwargs):
            for k in _GENERATE_LEAKED:
                kwargs.pop(k, None)
            return _orig_whisper_generate(self, *args, **kwargs)

        _whisper_forward_peft_compat._peft_compat_patched = True
        _whisper_generate_peft_compat._peft_compat_patched = True
        WhisperForConditionalGeneration.forward = _whisper_forward_peft_compat
        WhisperForConditionalGeneration.generate = _whisper_generate_peft_compat

    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb,
        attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
        device_map="auto",
    )
    model.config.suppress_tokens = []
    model.generation_config.language = cfg.get("language", "arabic")
    model.generation_config.task = cfg.get("task", "transcribe")
    model.generation_config.forced_decoder_ids = None

    # peft >= 0.13 enables gradient_checkpointing by default which roughly halves
    # train throughput. Honor the YAML's gradient_checkpointing flag instead.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", False)
    )

    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = _prepare_dataset(
        processor,
        train_jsonl=cfg["train_jsonl"],
        val_jsonl=cfg["val_jsonl"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        max_steps=cfg.get("max_steps", -1),
        bf16=True,
        fp16=False,
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        optim=cfg.get("optim", "paged_adamw_8bit"),
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 500),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 3),
        logging_steps=cfg.get("logging_steps", 25),
        predict_with_generate=True,
        generation_max_length=cfg.get("generation_max_length", 225),
        report_to=[] if args.no_wandb else cfg.get("report_to", ["tensorboard", "wandb"]),
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        seed=cfg.get("seed", 42),
        run_name=cfg.get("run_name"),
        push_to_hub=cfg.get("push_to_hub", False),
        hub_model_id=cfg.get("hub_model_id"),
        hub_strategy=cfg.get("hub_strategy", "every_save"),
        hub_private_repo=cfg.get("hub_private_repo", False),
    )

    if not args.no_wandb and "wandb" in (cfg.get("report_to") or []):
        import wandb

        wandb.init(
            project=cfg.get("wandb_project", "whisper-arabic-ft"),
            name=cfg.get("run_name"),
            tags=cfg.get("wandb_tags", []),
            config={
                **cfg,
                "normalizer_version": NORMALIZER_VERSION,
                "git_commit": _git_commit(),
            },
        )

    # transformers ≥4.50 renamed Trainer's `tokenizer=` kwarg to `processing_class=`;
    # we pass the feature extractor either way (Whisper's "tokenizer" for inputs is the
    # feature extractor, since the textual tokenizer is only used for label decoding).
    import inspect
    _trainer_kwargs = dict(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=WhisperDataCollator(processor=processor),
        compute_metrics=build_compute_metrics(processor),
    )
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters:
        _trainer_kwargs["processing_class"] = processor.feature_extractor
    else:
        _trainer_kwargs["tokenizer"] = processor.feature_extractor
    trainer = Seq2SeqTrainer(**_trainer_kwargs)

    # Optional early stopping on val WER plateau. Activates when
    # `early_stopping_patience` is set in the YAML config; off by default for
    # back-compat with the original 3-epoch fixed-length runs.
    es_patience = cfg.get("early_stopping_patience")
    if es_patience is not None:
        from transformers import EarlyStoppingCallback
        es_threshold = cfg.get("early_stopping_threshold", 0.001)  # ~0.1pp WER
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=int(es_patience),
                early_stopping_threshold=float(es_threshold),
            )
        )
        print(
            f"[train] EarlyStoppingCallback active: patience={es_patience}, "
            f"threshold={es_threshold}"
        )

    train_result = trainer.train()
    trainer.save_model(cfg["output_dir"])
    processor.save_pretrained(cfg["output_dir"])

    metrics = train_result.metrics
    metrics["normalizer_version"] = NORMALIZER_VERSION
    metrics["git_commit"] = _git_commit()
    Path(cfg["output_dir"], "train_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

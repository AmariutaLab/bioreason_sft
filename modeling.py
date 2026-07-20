"""Model loading helpers with a Conda-friendly Transformers default.

Unsloth is useful when available, but it is not available through Conda on many
clusters. These helpers keep the training scripts runnable with standard Conda
packages: transformers, peft, trl, pytorch, and bitsandbytes.
"""
from __future__ import annotations

from importlib import metadata

from packaging.version import Version
import torch


def disable_incompatible_torchao_for_peft():
    """Make PEFT ignore old TorchAO builds that its dispatcher cannot support."""
    try:
        torchao_version = Version(metadata.version("torchao").split("+", 1)[0])
    except metadata.PackageNotFoundError:
        return
    except Exception as exc:
        print(f"[model] could not inspect torchao version ({exc}); leaving PEFT unchanged")
        return

    if torchao_version >= Version("0.16.0"):
        return

    print(f"[model] torchao {torchao_version} is too old for this PEFT build; "
          "disabling PEFT TorchAO LoRA dispatch")
    try:
        import peft.import_utils as peft_import_utils

        peft_import_utils.is_torchao_available.cache_clear()
        peft_import_utils.is_torchao_available = lambda: False
    except Exception as exc:
        print(f"[model] could not patch peft.import_utils torchao check ({exc})")

    try:
        import peft.tuners.lora.torchao as peft_lora_torchao

        peft_lora_torchao.is_torchao_available = lambda: False
    except Exception:
        # The LoRA torchao dispatcher may not have been imported yet. In that
        # case patching peft.import_utils is sufficient for the later import.
        pass


def _want_unsloth(model_cfg) -> bool:
    backend = str(getattr(model_cfg, "backend", "auto")).lower()
    if backend == "transformers":
        return False
    if backend == "unsloth":
        return True
    return str(model_cfg.name).startswith("unsloth/")


def load_model_and_tokenizer(model_cfg):
    """Return (model, tokenizer, backend)."""
    if _want_unsloth(model_cfg):
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_cfg.name,
                max_seq_length=model_cfg.max_seq_length,
                load_in_4bit=model_cfg.load_in_4bit,
                dtype=None,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer, "unsloth"
        except ImportError:
            if str(getattr(model_cfg, "backend", "auto")).lower() == "unsloth":
                raise
            print("[model] unsloth not installed; falling back to transformers")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if bool(model_cfg.load_in_4bit):
        from transformers import BitsAndBytesConfig

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_cfg.name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, "transformers"


def add_lora(model, lora_cfg, seed: int, backend: str):
    if backend == "unsloth":
        from unsloth import FastLanguageModel

        return FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=list(lora_cfg.target_modules),
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )

    disable_incompatible_torchao_for_peft()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    peft_cfg = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=list(lora_cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_cfg)


def responses_only_trainer(trainer, instruction_part: str, response_part: str):
    try:
        from unsloth.chat_templates import train_on_responses_only

        return train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )
    except ImportError:
        print("[model] unsloth response-only masking unavailable; training on full text")
        return trainer


def prepare_for_inference(model, backend: str):
    if backend == "unsloth":
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(model)
    return model

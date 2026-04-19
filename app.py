import argparse
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS

# =============================
# Runtime / Device
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")

REPO_ID = "NAMAA-Space/NAMAA-Egyptian-TTS"
MODEL_FILES = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]

DEFAULT_EXAGGERATION = 0.5
DEFAULT_CFG_PACE = 0.5
DEFAULT_SEED = 0
DEFAULT_TEMPERATURE = 0.8

# =============================
# Global Model
# =============================
MODEL = None

# =============================
# Egyptian Arabic Defaults
# =============================
LANGUAGE_CONFIG = {
    "ar": {
        "audio": "./female_voice.wav",
        "text": "انا سبت الشغل و راجع دلوقتي علي طول.",
    }
}

EGYPTIAN_EXAMPLES = [
    "امبارح كنت بتفرج على فيلم، وفجأة الكهربا قطعت.",
    "أول ما وصلت البيت، غيرت هدومي وقعدت أرتاح شوية.",
    "مش كل مرة الحاجة تمشي زي ما إحنا مخططين.",
    "الجو حر قوي.",
    "اللي فات كان حاجة ودلوقتي حاجة تانية خالص.",
]


# =============================
# Helpers
# =============================
def default_audio_for_ui(lang: str) -> str | None:
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")


def default_text_for_ui(lang: str) -> str:
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")


def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("Loading model using from_pretrained...")
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=MODEL_FILES,
                token=os.getenv("HF_TOKEN"),
            )
        )

        MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)

        t3_path = os.path.join(ckpt_dir, "t3_mtl23ls_v2.safetensors")
        print(f"Loading t3 state from: {t3_path}")
        t3_state = load_safetensors(t3_path, device=DEVICE)
        MODEL.t3.load_state_dict(t3_state)
        MODEL.t3.to(DEVICE).eval()

        if hasattr(MODEL, "to") and str(getattr(MODEL, "device", "")) != DEVICE:
            MODEL.to(DEVICE)

        print("Model loaded successfully!")
    return MODEL


def set_seed(seed: int):
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def pick_random_egyptian_example() -> str:
    return random.choice(EGYPTIAN_EXAMPLES)


def validate_generation_inputs(
    text_input: str,
    exaggeration: float,
    cfg_pace: float,
    seed: int,
    temperature: float,
):
    if not text_input.strip():
        raise ValueError("الرجاء إدخال نص.")
    if not (0.25 <= float(exaggeration) <= 2.0):
        raise ValueError("قيمة exaggeration لازم تكون بين 0.25 و 2.0")
    if not (0.0 <= float(cfg_pace) <= 1.0):
        raise ValueError("قيمة cfg_pace لازم تكون بين 0.0 و 1.0")
    if not (0.05 <= float(temperature) <= 5.0):
        raise ValueError("قيمة temperature لازم تكون بين 0.05 و 5.0")
    if int(seed) < 0:
        raise ValueError("قيمة seed لازم تكون 0 أو رقم موجب")


def normalize_reference_path(reference_value: str | None) -> str | None:
    value = (reference_value or "").strip()
    if not value:
        return None
    reference_path = Path(value)
    if not reference_path.exists():
        raise FileNotFoundError(f"ملف الريفرنس غير موجود: {reference_path}")
    return str(reference_path)


def resolve_unique_output_path(
    output_name: str,
    output_dir: Path,
    default_stem: str = "output",
) -> Path:
    raw_name = (output_name or "").strip()
    if not raw_name:
        raw_name = default_stem

    raw_path = Path(raw_name)
    stem = raw_path.stem if raw_path.suffix else raw_path.name
    suffix = raw_path.suffix or ".wav"
    if not stem:
        stem = default_stem

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / f"{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = output_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def prompt_non_empty_text(prompt_label: str) -> str:
    while True:
        value = input(prompt_label).strip()
        if value:
            return value
        print("القيمة مطلوبة.")


def prompt_optional_reference() -> str | None:
    while True:
        raw_value = input("Reference audio path (اختياري - اضغط Enter للتخطي): ").strip()
        if not raw_value:
            return None
        candidate = Path(raw_value)
        if candidate.exists():
            return str(candidate)
        print("المسار غير موجود. حاول مرة تانية.")


def prompt_float_with_default(label: str, default: float, min_value: float, max_value: float) -> float:
    while True:
        raw_value = input(f"{label} [{default}]: ").strip()
        if not raw_value:
            return default
        try:
            value = float(raw_value)
        except ValueError:
            print("من فضلك أدخل رقم صحيح.")
            continue
        if min_value <= value <= max_value:
            return value
        print(f"القيمة لازم تكون بين {min_value} و {max_value}.")


def prompt_int_with_default(label: str, default: int, min_value: int = 0) -> int:
    while True:
        raw_value = input(f"{label} [{default}]: ").strip()
        if not raw_value:
            return default
        try:
            value = int(raw_value)
        except ValueError:
            print("من فضلك أدخل رقم صحيح.")
            continue
        if value >= min_value:
            return value
        print(f"القيمة لازم تكون اكبر من أو تساوي {min_value}.")


# =============================
# TTS
# =============================
def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str | None = None,
    exaggeration_input: float = DEFAULT_EXAGGERATION,
    temperature_input: float = DEFAULT_TEMPERATURE,
    seed_num_input: int = DEFAULT_SEED,
    cfgw_input: float = DEFAULT_CFG_PACE,
) -> tuple[int, np.ndarray]:
    validate_generation_inputs(
        text_input=text_input,
        exaggeration=exaggeration_input,
        cfg_pace=cfgw_input,
        seed=seed_num_input,
        temperature=temperature_input,
    )

    current_model = get_or_load_model()

    if seed_num_input and int(seed_num_input) != 0:
        set_seed(int(seed_num_input))

    chosen_prompt = audio_prompt_path_input or default_audio_for_ui("ar")

    generate_kwargs = {
        "exaggeration": float(exaggeration_input),
        "temperature": float(temperature_input),
        "cfg_weight": float(cfgw_input),
    }

    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt

    wav = current_model.generate(
        text_input[:300],
        language_id="ar",
        **generate_kwargs,
    )

    return (current_model.sr, wav.squeeze(0).numpy())


def resolve_output_path_from_args(args: argparse.Namespace) -> Path:
    explicit_output = (args.output or "").strip()
    if explicit_output:
        explicit_path = Path(explicit_output)
        output_dir = explicit_path.parent if str(explicit_path.parent) != "" else Path(".")
        return resolve_unique_output_path(
            output_name=explicit_path.name,
            output_dir=output_dir,
            default_stem="output",
        )

    output_dir = Path(args.output_dir)
    return resolve_unique_output_path(
        output_name=args.outputname,
        output_dir=output_dir,
        default_stem="output",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Egyptian Arabic speech with NAMAA TTS (session-aware CLI)."
    )

    parser.add_argument("text", nargs="?", default="", help="Text to synthesize (legacy mode).")

    parser.add_argument("--new", action="store_true", help="Start a new one-shot generation session.")
    parser.add_argument(
        "-p",
        "--prompt",
        "--PROMPT",
        dest="prompt",
        default=None,
        help="Prompt text for direct session mode.",
    )
    parser.add_argument(
        "-r",
        "--reference",
        "--audio-prompt",
        "--REFERENCE",
        dest="audio_prompt",
        default=None,
        help="Optional reference audio path.",
    )

    parser.add_argument(
        "--outputname",
        "--output-name",
        "--OUTPUTNAME",
        default="output",
        help="Base output name. If exists, a numbered name is generated.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for generated audio when using --outputname.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Explicit output file path (overrides --outputname, still avoids overwrite).",
    )

    parser.add_argument(
        "--exaggeration",
        "--EXAGGERATION",
        type=float,
        default=DEFAULT_EXAGGERATION,
        help="Speech expressiveness level.",
    )
    parser.add_argument(
        "--cfg-pace",
        "--cfg-weight",
        "--CFG_PACE",
        type=float,
        default=DEFAULT_CFG_PACE,
        help="CFG / pace value.",
    )
    parser.add_argument(
        "--seed",
        "--SEED",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed. Use 0 for stochastic behavior.",
    )
    parser.add_argument(
        "--temperature",
        "--TEMPERATURE",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )

    parser.add_argument("--use-example", action="store_true", help="Use a random Egyptian example sentence.")

    return parser.parse_args()


def build_session_inputs(args: argparse.Namespace) -> tuple[str, str | None, float, float, int, float]:
    if args.new and not args.prompt:
        print("Starting new interactive session...")
        text_input = prompt_non_empty_text("اكتب النص المطلوب تحويله لصوت: ")
        audio_prompt = prompt_optional_reference()
        exaggeration = prompt_float_with_default("EXAGGERATION", args.exaggeration, 0.25, 2.0)
        cfg_pace = prompt_float_with_default("CFG_PACE", args.cfg_pace, 0.0, 1.0)
        seed = prompt_int_with_default("SEED", args.seed, 0)
        temperature = prompt_float_with_default("TEMPERATURE", args.temperature, 0.05, 5.0)
        return text_input, audio_prompt, exaggeration, cfg_pace, seed, temperature

    if args.new:
        text_input = (args.prompt or "").strip()
        if not text_input:
            raise ValueError("في وضع --new المباشر لازم تمرر --prompt")
        return text_input, args.audio_prompt, args.exaggeration, args.cfg_pace, args.seed, args.temperature

    text_input = (args.prompt or args.text or "").strip()
    if args.use_example:
        text_input = pick_random_egyptian_example()
    elif not text_input:
        text_input = default_text_for_ui("ar")

    return text_input, args.audio_prompt, args.exaggeration, args.cfg_pace, args.seed, args.temperature


def main():
    args = parse_args()

    text_input, audio_prompt_raw, exaggeration, cfg_pace, seed, temperature = build_session_inputs(args)
    audio_prompt = normalize_reference_path(audio_prompt_raw)

    sample_rate, audio = generate_tts_audio(
        text_input=text_input,
        audio_prompt_path_input=audio_prompt,
        exaggeration_input=exaggeration,
        temperature_input=temperature,
        seed_num_input=seed,
        cfgw_input=cfg_pace,
    )

    output_path = resolve_output_path_from_args(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)

    print(f"Saved audio to: {output_path}")
    print(
        "Session summary -> "
        f"prompt_len={len(text_input)}, reference={'yes' if audio_prompt else 'no'}, "
        f"exaggeration={exaggeration}, cfg_pace={cfg_pace}, seed={seed}, temperature={temperature}"
    )


if __name__ == "__main__":
    main()
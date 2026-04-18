import argparse
import random
import numpy as np
import torch
from pathlib import Path
import os

from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors
import soundfile as sf

from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# =============================
# Runtime / Device
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on device: {DEVICE}")

REPO_ID = "NAMAA-Space/NAMAA-Egyptian-TTS"
MODEL_FILES = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]

# =============================
# Global Model
# =============================
MODEL = None

# =============================
# Egyptian Arabic UI Config 🇪🇬
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
        # Load base model
        MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
        
        # Load and replace t3 state from safetensors
        t3_path = os.path.join(ckpt_dir, "t3_mtl23ls_v2.safetensors")
        print(f"Loading t3 state from: {t3_path}")
        t3_state = load_safetensors(t3_path, device=DEVICE)
        MODEL.t3.load_state_dict(t3_state)
        MODEL.t3.to(DEVICE).eval()
        
        # Ensure entire model is on correct device
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

# =============================
# TTS
# =============================
def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
) -> tuple[int, np.ndarray]:

    current_model = get_or_load_model()

    if seed_num_input and int(seed_num_input) != 0:
        set_seed(int(seed_num_input))

    text_input = (text_input or "").strip()
    if not text_input:
        raise ValueError("الرجاء إدخال نص.")

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

def pick_random_egyptian_example():
    return random.choice(EGYPTIAN_EXAMPLES)

def main():
    parser = argparse.ArgumentParser(description="Generate Egyptian Arabic speech with NAMAA TTS.")
    parser.add_argument("text", nargs="?", default="", help="Text to synthesize.")
    parser.add_argument("--audio-prompt", dest="audio_prompt", default=None, help="Optional reference audio path.")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Speech expressiveness level.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Use 0 to keep stochastic behavior.")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG / pace weight.")
    parser.add_argument("--output", default="output.wav", help="Output WAV file path.")
    parser.add_argument("--use-example", action="store_true", help="Use a random Egyptian example sentence.")
    args = parser.parse_args()

    text_input = args.text.strip()
    if args.use_example:
        text_input = pick_random_egyptian_example()
    elif not text_input:
        text_input = default_text_for_ui("ar")

    sample_rate, audio = generate_tts_audio(
        text_input,
        audio_prompt_path_input=args.audio_prompt,
        exaggeration_input=args.exaggeration,
        temperature_input=args.temperature,
        seed_num_input=args.seed,
        cfgw_input=args.cfg_weight,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    print(f"Saved audio to: {output_path}")


if __name__ == "__main__":
    main()
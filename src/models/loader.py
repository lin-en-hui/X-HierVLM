import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
import open_clip

from config.settings import *

_VLM_MODEL = None
_VLM_PROCESSOR = None
_BIOCLIP_MODEL = None
_BIOCLIP_PREPROCESS = None
_BIOCLIP_TOKENIZER = None

def load_vlm():
    """Load Qwen2.5-VL base model with LoRA adapter (lazy loading)."""
    global _VLM_MODEL, _VLM_PROCESSOR
    if _VLM_MODEL is None:
        print("Loading Qwen2.5-VL base model...")
        processor = AutoProcessor.from_pretrained(VLM_MODEL_PATH)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_PATH,
            device_map={"": DEVICE},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",  # optional, remove if not available
        )
        print(f"Loading LoRA adapter from {LORA_PATH}...")
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        model.eval()
        _VLM_MODEL = model
        _VLM_PROCESSOR = processor
    return _VLM_MODEL, _VLM_PROCESSOR

def load_bioclip(species_list):
    """
    Load BioCLIP 2.5 model and precompute text features for given species list.
    Returns (model, preprocess_fn, text_features_tensor).
    """
    global _BIOCLIP_MODEL, _BIOCLIP_PREPROCESS, _BIOCLIP_TOKENIZER
    if _BIOCLIP_MODEL is None:
        print("Loading BioCLIP 2.5 model...")
        model, preprocess, _ = open_clip.create_model_and_transforms(
            BIOCLIP_MODEL_NAME,
            pretrained=None,  # uses default pretrained weights from hub
            device=DEVICE,
            precision='bf16'
        )
        tokenizer = open_clip.get_tokenizer(BIOCLIP_MODEL_NAME)
        model.eval()
        _BIOCLIP_MODEL = model
        _BIOCLIP_PREPROCESS = preprocess
        _BIOCLIP_TOKENIZER = tokenizer

    # Precompute text features for the given species list
    prompts = [f"a photo of a {sp}, a type of animal." for sp in species_list]
    text_tokens = _BIOCLIP_TOKENIZER(prompts).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        text_features = _BIOCLIP_MODEL.encode_text(text_tokens)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
    return _BIOCLIP_MODEL, _BIOCLIP_PREPROCESS, text_features
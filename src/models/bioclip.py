import torch
import torch.nn.functional as F
from PIL import Image
from models.loader import load_bioclip

def predict_bioclip(image_path, species_list):
    """
    Perform black-box prediction using BioCLIP.
    Returns (predicted_species, confidence_score).
    """
    model, preprocess, text_features = load_bioclip(species_list)
    device = text_features.device

    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            image_features = model.encode_image(image_input)
            image_features = F.normalize(image_features, dim=-1)
            similarity = 100.0 * image_features @ text_features.T
            probs = similarity.softmax(dim=-1)
        top_prob, top_idx = probs[0].max(dim=0)
    return species_list[top_idx.item()], top_prob.item()
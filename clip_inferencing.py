import torch
import torch.nn.functional as F
import pickle

from clip_model import CLIPModel
from configuration import CFG

def load_model(model_path):
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    return model

def load_df():
    with open("pickles/valid_df.pkl", 'rb') as file:
        valid_df = pickle.load(file)
        return valid_df

def load_image_embeddings():
    with open("pickles/image_embeddings.pkl", 'rb') as file:
        image_embeddings = pickle.load(file)
        return image_embeddings
    
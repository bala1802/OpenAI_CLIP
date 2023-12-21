import gradio as gr
import torch

import torch.nn.functional as F
from transformers import DistilBertTokenizer

from PIL import Image
import numpy as np
import requests

import clip_inferencing as inference

device="cpu"
valid_df = inference.load_df()
image_embeddings = inference.load_image_embeddings()
model = inference.load_model(model_path="model/best.pt")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)

n=9
image_filenames=valid_df['image'].values

with gr.Blocks() as demo:
    
    def inference(query):
        encoded_query = tokenizer([query])
        batch = {
                    key: torch.tensor(values).to(device)
                    for key, values in encoded_query.items()
                }
        with torch.no_grad():
            text_features = model.text_encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            text_embeddings = model.text_projection(text_features)
        
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = text_embeddings_n @ image_embeddings_n.T

        values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
        matches = [image_filenames[idx] for idx in indices[::5]]

        resulting_images = []
        for match in matches:
            img_https_link = "https://raw.githubusercontent.com/bala1802/ERA_Session19/main/Images/" + match
            resulting_images.append(np.array(Image.open(requests.get(img_https_link, stream=True).raw).convert('RGB')))

            # resulting_images.append(np.array(Image.open(f"Images/{match}").convert('RGB')))
        return resulting_images
    
    gr.Markdown(
                """
                    # CLIP Demo !!!
                """
                )
    with gr.Column(variant="panel"):
        with gr.Row():
            text = gr.Textbox(
                label="Enter your prompt",
                max_lines=1,
                placeholder="Extract the matching images ....",
                container=False,
            )
            btn = gr.Button("Show Images", scale=0)

        gallery = gr.Gallery(
            label="Movies", show_label=False, elem_id="gallery"
        , columns=[4], rows=[1], object_fit="contain", height="auto")
    
    btn.click(inference, text, gallery)

if __name__ == "__main__":
    demo.launch(share=True)
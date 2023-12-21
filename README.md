# Introduction

The objective of this repository is to implement Open AI's CLIP Paper: _Learning Transferable Visual Models From Natural Language Supervision_ from scratch using PyTorch.

Read Paper: https://arxiv.org/pdf/2103.00020.pdf

# About CLIP

<img width="1209" alt="image" src="https://github.com/bala1802/OpenAI_CLIP/assets/22103095/c3a6fd3e-962c-409d-8f3f-1a657a2933ba">


- A model designed for learning joint representations of images and text.
- Leverages a shared embedding space, where images and their corresponding textual descriptions are mapped to similar points.
- Uses a contrastive learning objective to train the model. It aims to maximize the similarity between positive pairs (Correct Image-Text pairs) and minimize the similarity between negative pairs (incorrect pairs)

# Code Implementation

## Text Encoder

- The _distilbert-base-uncased_ encoder model for embedding the texts.
- The resulting text encoder embedding dimension will be of shape - `(batch_size, text_embedding)` -> `(32, 768)`

## Image Encoder

- The _resnet50_ model pretrained model is used for encoding the images
- The resulting image encoder embedding dimension will be of shape - `(batch_size, image_embedding)` -> `(32, 2048)`

## Projection Head

The Projection Head serves a crucial role in shaping the representations learned by the model.

- 



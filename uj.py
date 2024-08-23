import ssl
import certifi
import torch
import clip
from PIL import Image
import torch.nn.functional as F

ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: ssl_context

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
def get_embedding(image_path):
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
    return embedding
def compare_embeddings(embedding1, embedding2):
    similarity = F.cosine_similarity(embedding1, embedding2)
    print(f"Benzerlik Skoru: {similarity.item()}")
    if similarity.item() > 0.9:
        print("Nesneler aynı.")
    else:
        print("Nesneler farklı.")

image_path1 = "/---/----/---/---/---.jpg"
image_path2 = "/---/----/---/---/---.jpg"

embedding1 = get_embedding(image_path1)
embedding2 = get_embedding(image_path2)

compare_embeddings(embedding1, embedding2)

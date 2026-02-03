from PIL import Image
import clip
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
model.eval()

img = Image.open("face.jpg")  # take a selfie & save it
image = preprocess(img).unsqueeze(0).to(device)

text = clip.tokenize([
    "a person",
    "a human face",
    "a dog",
    "a chair",
]).to(device)

with torch.no_grad():
    img_feat = model.encode_image(image)
    img_feat /= img_feat.norm(dim=-1, keepdim=True)

    txt_feat = model.encode_text(text)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

scores = (img_feat @ txt_feat.T).squeeze()
print(scores)

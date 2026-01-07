import torch
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id2label = {0: "NonPolitical", 1: "Political"}

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)

model.load_state_dict(torch.load("model/vit_meme_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def predict(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image).logits
        pred = torch.argmax(logits, dim=1).item()
    return id2label[pred]

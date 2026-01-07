import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, AdamW
from tqdm import tqdm

# Configuration
MODEL_NAME = "google/vit-base-patch16-224"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
NUM_LABELS = 2
OUTPUT_DIR = "model"
MODEL_FILE = "vit_meme_model.pth"

def train(data_dir, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Load Dataset
    print(f"Loading data from: {data_dir}")
    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    except FileNotFoundError:
        print(f"Error: Dataset directory '{data_dir}' not found.")
        return

    # Check for empty folders or invalid structure
    if len(dataset.classes) != 2:
        print(f"Error: Expected 2 classes (Political, NonPolitical), found {len(dataset.classes)}: {dataset.classes}")
        return

    # Split Data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize Model
    print("Initializing ViT model...")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Training Loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0
        
        # Training
        for batch in tqdm(train_loader, desc="Training"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Average Training Loss: {train_loss / len(train_loader):.4f}")
        
        # Validation
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                preds = torch.argmax(outputs.logits, dim=1)
                val_acc += (preds == labels).sum().item()
        
        val_accuracy = val_acc / len(val_dataset)
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save Model
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    save_path = os.path.join(OUTPUT_DIR, MODEL_FILE)
    print(f"Saving model to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT for Meme Classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory containing class folders")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    
    train(args.data_dir, args.epochs, args.batch_size)

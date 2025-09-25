# 🌱 Pretrained Latent Diffusion Models for Plant Leaf Augmentation

We propose **pretrained models of Latent Diffusion** trained from scratch for mango leaf disease images.  
These models can be used to **augment any other plant leaf datasets** by generating synthetic but realistic images.  
Researchers and readers can fine-tune these pretrained models for their own datasets and generate new images.

---

📂 Pretrained Model Files
Our pretrained model consists of:
- encoder.pth  
- decoder.pth  
- unet.pth  

All trained up to 1000 epochs on mango leaf disease dataset.
---
How to Use

1️⃣ Setup Environment

git clone https://github.com/pankajgithup2025-cmd/PreTrainedLeafDiffusion.git
cd PreTrainedLeafDiffusion

# (Optional) create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

# install required dependencies
pip install -r requirements.txt

import torch

# Example: loading pretrained components
encoder = torch.load("encoder.pth", map_location="cpu")
decoder = torch.load("decoder.pth", map_location="cpu")
unet    = torch.load("unet.pth", map_location="cpu")
3️⃣ Fine-tuning on Your Dataset
Prepare your own plant leaf dataset (RGB images, recommended size 512×512).
Use the pretrained models as initialization:
Encoder + Decoder for latent space transformation
U-Net for diffusion learning
Train for fewer epochs (e.g., 100–200) since base model is already trained.
# pseudo-code
train(model=unet, data=my_leaf_dataset, epochs=200, lr=1e-5)

4️⃣ Generate New Images
After fine-tuning, you can sample synthetic images:
# pseudo-code for generation
images = diffusion_sampling(encoder, decoder, unet, num_samples=50)
save_images(images, "augmented/")


Applications

1.Data augmentation for plant disease classification 

2.Extending limited datasets into large-scale datasets

3.Benchmarking generative models in agriculture

🔹 1. train.py (fine-tuning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os

# ----------------------------
#  Dataset Loader (example: folder with leaf images)
# ----------------------------
def get_dataloader(data_dir, img_size=512, batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ----------------------------
#  Load pretrained components
# ----------------------------
def load_models():
    encoder = torch.load("encoder.pth", map_location="cpu")
    decoder = torch.load("decoder.pth", map_location="cpu")
    unet    = torch.load("unet.pth", map_location="cpu")
    return encoder, decoder, unet

# ----------------------------
#  Training loop (simplified)
# ----------------------------
def train_model(data_dir, epochs=100, lr=1e-5, save_path="finetuned_unet.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, decoder, unet = load_models()
    unet.to(device)

    dataloader = get_dataloader(data_dir)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(unet.parameters(), lr=lr)

    print("🚀 Starting fine-tuning...")

    for epoch in range(epochs):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # Encode → add noise → denoise with UNet → Decode
            z = encoder(images)
            noisy_z = z + 0.1 * torch.randn_like(z)
            denoised_z = unet(noisy_z)
            recon = decoder(denoised_z)

            loss = criterion(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    torch.save(unet.state_dict(), save_path)
    print(f"✅ Fine-tuned UNet saved at {save_path}")

if __name__ == "__main__":
    # Example usage: put your dataset inside ./data/my_leaf_dataset
    train_model(data_dir="./data/my_leaf_dataset", epochs=50)



If you use this pretrained model in your work, please cite this repository.

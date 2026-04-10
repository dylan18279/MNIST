import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

# --------------------------
# 1️⃣ Data (MNIST)
# --------------------------
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform) # change to where your dataset is or switch to download=True
test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform) # same

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --------------------------
# 2️⃣ CNN for MNIST (FIXED)
# --------------------------
class AlexNetMNIST(nn.Module):
    def __init__(self):
        super(AlexNetMNIST, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                # 14x14
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 7x7
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)                 # 3x3
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),   # FIXED size
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --------------------------
# 3️⃣ Training
# --------------------------
def train_model(model, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    start_total = time.time()
    
    for epoch in range(epochs):
        start_epoch = time.time()
        total_loss = 0
        
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Time: {time.time()-start_epoch:.2f}s")
    
    print(f"Total Training Time: {time.time()-start_total:.2f}s")

# --------------------------
# 4️⃣ Save test images
# --------------------------
def save_test_images_with_labels(model, output_dir='test_images'):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    correct, total = 0, 0
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            for i in range(len(labels)):
                img = images[i].cpu().squeeze().numpy()
                label = labels[i].item()
                pred = preds[i].item()
                
                label_folder = os.path.join(output_dir, f'label_{label}')
                os.makedirs(label_folder, exist_ok=True)
                
                fig, ax = plt.subplots(figsize=(2,2))
                ax.imshow(img, cmap='gray')
                ax.set_title(f"T:{label} P:{pred}", fontsize=8)
                ax.axis('off')
                
                save_path = os.path.join(label_folder, f'img_{batch_idx}_{i}.png')
                fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)
    
    print(f"Test Accuracy: {correct/total*100:.2f}%")

# --------------------------
# 5️⃣ NEW: Test 2 custom images
# --------------------------
def test_two_images(model, image_paths):
    assert len(image_paths) == 2, "Provide exactly 2 images"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path)
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            pred = model(input_tensor).argmax(dim=1).item()
            
            plt.imshow(img, cmap='gray')
            plt.title(f"Prediction: {pred}")
            plt.axis('off')
            plt.show()

# --------------------------
# 6️⃣ Run
# --------------------------
alexnet = AlexNetMNIST()

train_model(alexnet, epochs=5)

# 🔥 YOUR CUSTOM TEST
test_two_images(alexnet, [
    "0_digit.png",
    "1_digit.png"
])
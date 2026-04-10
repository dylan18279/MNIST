import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time

# Transform: convert to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SmartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train_model(model, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    start_total = time.time()  # Total training timer
    
    for epoch in range(epochs):
        start_epoch = time.time()  # Epoch timer
        total_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        end_epoch = time.time()
        epoch_time = end_epoch - start_epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Time: {epoch_time:.2f}s")
    
    end_total = time.time()
    total_time = end_total - start_total
    print(f"Total Training Time: {total_time:.2f}s")
    
def test_model(model, n_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Test Accuracy: {correct/total*100:.2f}%")
    
    # Show predictions for first n_samples images
    for i in range(n_samples):
        img = test_dataset[i][0].squeeze()
        label = test_dataset[i][1]
        pred = model(test_dataset[i][0].unsqueeze(0).to(device)).argmax().item()
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label}, Predicted: {pred}")
        plt.show()

def show_mistakes_grid(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    mistakes = []

    # Collect all mistakes
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    mistakes.append((images[i].cpu().squeeze(), labels[i].item(), preds[i].item()))
    
    print(f"Total mistakes: {len(mistakes)}")

    if len(mistakes) == 0:
        print("No mistakes! Perfect model!")
        return

    # Grid size
    n = len(mistakes)
    cols = 10  # number of columns in the grid
    rows = math.ceil(n / cols)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
    axes = axes.flatten()
    
    # Plot all mistakes
    for i, ax in enumerate(axes):
        if i < n:
            img, label, pred = mistakes[i]
            ax.imshow(img, cmap='gray')
            ax.set_title(f"{label}->{pred}", fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


smart = SmartModel()
train_model(smart, epochs=5)
test_model(smart)
show_mistakes_grid(smart)
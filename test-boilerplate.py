# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import os

# # 1. Hyperparameters
# BATCH_SIZE = 64
# LR = 0.001
# EPOCHS = 10
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 2. Dataset (Example: CIFAR10)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
# test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# # 3. Model Definition
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SimpleCNN, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Flatten(),
#             nn.Linear(32 * 8 * 8, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         return self.net(x)

# model = SimpleCNN().to(DEVICE)

# # 4. Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)

# # 5. Training Loop
# def train(model, loader, criterion, optimizer):
#     model.train()
#     running_loss = 0.0
#     for inputs, targets in loader:
#         inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#     return running_loss / len(loader)

# # 6. Evaluation
# def evaluate(model, loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for inputs, targets in loader:
#             inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += targets.size(0)
#             correct += (predicted == targets).sum().item()
#     return 100 * correct / total

# # 7. Training Loop Execution
# for epoch in range(EPOCHS):
#     train_loss = train(model, train_loader, criterion, optimizer)
#     test_acc = evaluate(model, test_loader)
#     print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
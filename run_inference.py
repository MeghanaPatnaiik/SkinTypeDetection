import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import kagglehub
from PIL import Image

import download_models
download_models.download_models()

# ---------------------------
# 1. Download dataset
# ---------------------------
print(" Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("shakyadissanayake/oily-dry-and-normal-skin-types-dataset")
base_dir = os.path.join(path, "Oily-Dry-Skin-Types")

test_dir = os.path.join(base_dir, "test")
if not os.path.exists(test_dir):
    print(" Test folder not found. Please check dataset structure.")
else:
    print(f"Test folder found: {test_dir}")

# ---------------------------
# 2. Device setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ---------------------------
# 3. Preprocessing
# ---------------------------
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes


# ---------------------------
# 4. Function to Evaluate Model
# ---------------------------
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


# ---------------------------
# 5. Load BOTH Models (FIXED)
# ---------------------------

# ----- Load RESNET50 -----
print("\n Loading ResNet50 model...")

model_resnet50 = models.resnet50(pretrained=False)
num_ftrs_50 = model_resnet50.fc.in_features
model_resnet50.fc = nn.Linear(num_ftrs_50, 3)

state_dict_50 = torch.load("resnet50_skin_classifier.pth", map_location=device)
model_resnet50.load_state_dict(state_dict_50)
model_resnet50.to(device)
print(" ResNet50 loaded!")

# ----- Load RESNET152 -----
print("\n Loading ResNet152 model...")

model_resnet152 = models.resnet152(pretrained=False)
num_ftrs_152 = model_resnet152.fc.in_features
model_resnet152.fc = nn.Linear(num_ftrs_152, 3)

state_dict_152 = torch.load("resnet152_skin_classifier.pth", map_location=device)
model_resnet152.load_state_dict(state_dict_152)
model_resnet152.to(device)
print(" ResNet152 loaded!")


# ---------------------------
# 6. Evaluate BOTH Models
# ---------------------------
print("\nEvaluating ResNet50 on test set...")
acc_resnet50 = evaluate_model(model_resnet50, test_loader)
print(f" ResNet50 Test Accuracy: {acc_resnet50:.2f}%\n")

print("\nEvaluating ResNet152 on test set...")
acc_resnet152 = evaluate_model(model_resnet152, test_loader)
print(f" ResNet152 Test Accuracy: {acc_resnet152:.2f}%\n")


# ---------------------------
# 7. Example prediction
# ---------------------------
example_img = os.path.join(test_dir, class_names[0],
                           os.listdir(os.path.join(test_dir, class_names[0]))[0])
print(f"Example image used: {example_img}")

img = Image.open(example_img).convert("RGB")
img_t = data_transforms(img).unsqueeze(0).to(device)

# --- ResNet50 prediction ---
out50 = model_resnet50(img_t)
_, pred50 = torch.max(out50, 1)
print(f"\nResNet50 Prediction: {class_names[pred50.item()]}")

# --- ResNet152 prediction ---
out152 = model_resnet152(img_t)
_, pred152 = torch.max(out152, 1)
print(f"ResNet152 Prediction: {class_names[pred152.item()]}")

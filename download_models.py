import gdown
import os

# Model file names
MODELS = {
    "resnet50_skin_classifier.pth": "1kpHaH3FxWzAzA-GJUTfkpA7kfGAVFYgL",
    "resnet152_skin_classifier.pth": "1aM_e3TCmcZMA_4FHwpWEJZfj-Hje3azS"
}

def download_models():
    for model_name, file_id in MODELS.items():
        if not os.path.exists(model_name):
            print(f"Downloading {model_name}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_name, quiet=False)
            print(f"{model_name} downloaded successfully.\n")
        else:
            print(f"{model_name} already exists.\n")

if __name__ == "__main__":
    download_models()

# Skin Type Classification – ResNet50 & ResNet152

This project performs skin type classification (Oily / Dry / Normal) using ResNet50 and ResNet152 models trained on the Kaggle dataset "Oily, Dry and Normal Skin Types Dataset". 



---

##  1. Clone the Repository

```bash
git clone https://github.com/MeghanaPatnaiik/SkinTypeDetection.git
cd SkinTypeDetection
```

---

##  2. Create & Activate Virtual Environment

### Linux / Mac
```bash
python -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate 
```

---

##  3. Install Dependencies

```bash
pip install -r requirements.txt
```

---
## ▶️ 4.  Download the model

```bash
python download_models.py
```

## ▶️ 5. Run Inference & Evaluation

```bash
python run_inference.py
```

---

## 📊 Results

The script will output accuracy metrics comparing ResNet50 and ResNet152 performance on the test set.

---


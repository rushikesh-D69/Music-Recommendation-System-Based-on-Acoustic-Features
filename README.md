Here is a **complete, detailed, production-level README.md** that you can directly paste into your GitHub repository.
It includes description, architecture, installation, usage, screenshots section, model explanation, and contribution guidelines.

---

# Music Recommendation System Based on Acoustic Features

A deep-learning powered **content-based music recommendation system** that analyzes **acoustic features**—MFCCs, chroma vectors, tempo, BPM, and chord progressions—to recommend harmonically similar songs. Built using **Python, TensorFlow, Streamlit**, and **audio signal processing techniques**.

---

## Features

* Extracts rich audio features using MFCCs, Chroma STFT, Tempo/BPM, and Chord Progressions
* Converts feature vectors into **6×6×1 tensors** for CNN processing
* Deep learning model built using **Convolutional Neural Networks (CNN)**
* Recommends songs using **cosine similarity on learned embeddings**
* Includes interactive **Streamlit UI** for visualization:

  * Chord histograms
  * Tempo & MFCC graphs
  * Cosine similarity heatmaps
  * Audio playback
* Scalable & fast due to caching, batch processing, and lazy loading

---

## System Architecture

```
Audio Input → Feature Extraction → Feature Normalization → 6×6 Matrix Encoding
         → CNN Model → Embedding Vector → Cosine Similarity → Recommendations
```

---

## Tech Stack

* **Python**
* **TensorFlow/Keras**
* **NumPy, Librosa, Scikit-learn**
* **Streamlit** (Frontend)
* **Matplotlib / Seaborn** for visualizations
* **GitHub + Streamlit Cloud** for deployment

---

## 🛠 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Music-Recommendation-Acoustic.git
cd Music-Recommendation-Acoustic
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## Dataset & Feature Extraction

### Extract Features

```python
import librosa
import numpy as np

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, duration=30)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    mfcc_mean = np.mean(mfcc.T, axis=0)
    chroma_mean = np.mean(chroma.T, axis=0)

    feature_vector = np.concatenate([mfcc_mean, chroma_mean, [tempo]])
    return feature_vector
```

---

## CNN Model (Simplified Version)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(6, 6, 1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(12, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```

---

## Recommendation Engine

Uses **Cosine Similarity** between feature embeddings or predicted chord distributions.

```python
from sklearn.metrics.pairwise import cosine_similarity

def recommend(query_vec, database_vectors, top_k=5):
    similarities = cosine_similarity([query_vec], database_vectors)[0]
    indices = similarities.argsort()[-top_k:][::-1]
    return indices, similarities[indices]
```

---

## Streamlit UI

Run the Streamlit application to load your interactive music dashboard.

```bash
streamlit run app.py
```

### UI Features:

* Upload a song or select from dataset
* Visualize MFCCs, Chroma, Tempo, Chord Histogram
* View similarity heatmap
* Play audio preview
* Get top-5 recommended songs

---

## Results

| Metric                      | Value    |
| --------------------------- | -------- |
| Accuracy                    | **0.84** |
| F1-Score                    | **0.20** |
| Top-3 Accuracy              | **0.90** |
| Cosine Similarity Precision | **0.74** |

---

## Project Structure

```
│── app.py               # Streamlit Frontend
│── model/
│     ├── cnn_model.h5   # Saved Model
│     └── scaler.pkl     # StandardScaler
│── features/
│     └── extracted_features.npy
│── utils/
│     ├── extract.py     # Feature Extraction
│     └── recommend.py   # Recommendation Engine
│── screenshots/
│── requirements.txt
│── README.md
```

---

## Future Enhancements

* Add RNN/CRNN for sequential pattern modeling
* Add multimodal data (lyrics, metadata, mood)
* Support Spotify API integration
* Expand dataset for higher accuracy

---

## Contributors

* **Chandan Sai Pavan Padala**
* **D Rushikesh**

---

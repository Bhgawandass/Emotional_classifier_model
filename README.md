# Speech Emotion Recognition Web App using GRU + Attention

## 🔍 Project Overview

This project involves developing a deep learning-based system for recognizing emotions in speech. By extracting MFCC features from audio signals, it uses a GRU architecture enhanced with an attention mechanism to identify emotional states. The system accepts .wav audio files as input and is capable of detecting one of eight predefined emotions.

## 🎯 Goals
* Automatically recognize emotional states in speech recordings.
* Create an intuitive user interface for real-time inference.
* Deploy a fully functional app to the web using Streamlit Cloud.

## 🧠 Dataset: RAVDESS

* Source: [RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/record/1188976)
* Subset: Only **speech** files (Modality code = `01`) used.
* Classes: `angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised`

## 🧹 Data Preprocessing

1. **Audio loading**:Navigates RAVDESS directories to access .wav files and extract emotion metadata.
2. **Features Extraction**: Converts each audio clip into 40 MFCCs per frame using librosa.
3. **Normalization**:All audio features are padded or truncated to 174 frames for uniform input.
4. **Encoding**: Emotion labels are transformed to numeric format using scikit-learn’s LabelEncoder.

## 🧱 Model Architecture

* **Input**: `(174, 40)` MFCC features
* **GRU Layer**: Bidirectional GRU (128 units)
* **Attention Layer**: Attention Layer (custom)
* **Dense Outout**: Fully connected layer for classification
* **Activation**: `softmax`

> ✅ The model is trained using categorical cross-entropy and Adam optimizer.

## 📊 Performance Metrics

After tuning and training:

* ✅ **Macro F1 Score**: **84.32%**
* ✅ **Overall Accuracy**: **85%**
* Per-class metrics:

  * `angry`: F1 = 0.92
  * `calm`: F1 = 0.88
  * `happy`: F1 = 0.84
  * `surprised`: F1 = 0.91
  * `neutral`: F1 = 0.74

## 🌐 Web Application (Streamlit)

The app allows users to:

* Upload and preview .wav files
* Automatic MFCC extraction and prediction
* Instant emotion classification results

### Key Features

* Clean UI with `st.audio()` and emotion output
* Loads pre-trained GRU-Attention model and label encoder (`.h5` and `.pkl`)
* Uses `librosa` for MFCC extraction

## 🛠️ Setup & Deployment

### Local

```bash
# Clone repo
https://github.com/Bhgawandass/Emotional_classifier_model.git

# Create virtual env
python -m venv mars
source mars/bin/activate  # or .\mars\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

### Streamlit Cloud

* ✅ Python 3.9 runtime (set automatically by Streamlit)
* ✅ Ensure `emotion_gru_attention_model.h5` and `label_encoder.pkl` are in repo
* ✅ Correct `requirements.txt` (compatible with both `tensorflow==2.11.0` and `streamlit==1.25.0`)

## 📁 Directory Layout

```
emotion-classification-app/
├── app.py
├── emotion_gru_attention_model.h5
├── label_encoder.pkl
├── requirements.txt
├── test.py
└── README.md
```
##streamlit link
(https://emotionalclassifiermodel-9zs5eqjjklzasllvogsnrw.streamlit.app/)

## 💡 What's Next

* Integrate real-time recording via microphone input
* Add multilingual support
* Experiment with transformer-based models (e.g., Wav2Vec2.0, HuBERT)
* Expand training with more diverse datasets

---

**Author**: [Bhagwan Dass](https://github.com/Bhgawandass)

**License**: MIT
"# Emotional_classifier" 

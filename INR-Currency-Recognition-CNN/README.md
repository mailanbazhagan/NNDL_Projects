
# Indian Currency Denomination Detection

A deep learning project that detects Indian currency denominations from images using a Convolutional Neural Network (CNN) and spells out the detected denomination through audio output.  
This project is designed to assist **visually impaired individuals** and to provide an **automated way of recognizing Indian currency notes**.

---

## ✨ Features

- 🔍 Detects Indian currency notes: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000.
- 🧠 Uses a trained CNN model for classification.
- 🔊 Provides **audio feedback** of the detected denomination.
- ⚡ Lightweight and easy to deploy.

---

## 🛠 Tech Stack

- **Python 3**
- **TensorFlow** – Deep Learning framework
- **OpenCV** – Image preprocessing
- **Pyttsx3** – Text-to-speech conversion
- **NumPy** – Numerical operations
- **Matplotlib** – Visualization

---

## ⚙️ How It Works

1. **Image Input**  
   Capture or upload an image of an Indian currency note.

2. **Preprocessing**  
   - Resize the image to the model's input size.
   - Normalize pixel values.
   - Prepare the image for CNN input.

3. **Prediction**  
   - The trained CNN model classifies the note into one of the known denominations.

4. **Audio Output**  
   - The predicted denomination is converted into speech using Pyttsx3.
   - The audio is played aloud to announce the detected note.

---


---

## 🚀 Future Improvements (optional)

- Extend to detect **damaged** or **folded** notes.
- Mobile or web deployment for wider accessibility.
- Multi-language audio feedback support.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

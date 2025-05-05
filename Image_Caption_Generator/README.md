
# Image Caption Generator

## Problem Statement
The project aims to develop an Image Caption Generator using a **CNN** for feature extraction and an **LSTM** for text generation. The model is trained on the **Flickr8k** dataset to automatically generate descriptive captions for input images.

## Dataset Sources 
- **Flickr8k Dataset**  
  [https://www.kaggle.com/datasets/adityajn105/flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)  
- **Flickr30k Dataset**  
  [https://www.kaggle.com/datasets/adityajn105/flickr30k](https://www.kaggle.com/datasets/adityajn105/flickr30k)  
- **Tiny ImageNet Dataset**  
  [https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)  

## Team Members
- **Shreyas** (CB.EN.U4CSE22154)  
- **Uma Mahesh** (CB.EN.U4CSE22534)  
- **Sathvik Reddy** (CB.EN.U4CSE22160)  
- **Hemanth Reddy** (CB.EN.U4CSE22558)  

## Dataset Used
- 📁 **Tiny ImageNet-200**  
- 📁 **Flickr30k**  
- 📁 **Flickr8k**  

## Preprocessing of Data
1. **Image Processing**  
   - Resize images to uniform dimensions  
   - Convert to numerical arrays for CNN input  

2. **Caption Processing**  
   - Tokenize captions and convert words to integer sequences  
   - Pad sequences to a fixed length  
   - Generate word mappings and embeddings  

## Model Architecture & Design
1. **CNN (VGG16)**  
   - Pretrained VGG16 extracts high-level image features  
2. **Embedding Layer**  
   - Maps each word token to a dense vector representation  
3. **LSTM Network**  
   - Generates captions based on extracted features and previous word inputs  
4. **Dense Layers**  
   - Output layer predicts the next word in the sequence  

## Optimization & Hyperparameter Tuning
- **Optimizer:** Adam (and comparisons with SGD)  
- **Hyperparameters Tuned:**  
  - Learning rate  
  - Batch size  
  - Sequence length  
  - Dropout rate  
- Validation loss used to monitor overfitting  

## Evaluation Metrics
- **BLEU Score**  
- **RMSE**  
- **MSE**  
- **Accuracy**  
- **Loss**  

## Trained Models
All trained models are stored on Google Drive and can be accessed here:  

🔗 [VGG16_Tiny_Imagenet_model](https://drive.google.com/file/d/1anfxA4Fg_2rWkWZhnyeZpFaemIaFShwV/view?usp=sharing)  

🔗 [features.pkl file](https://drive.google.com/file/d/1aEdbZ7ezSuwr-aN-n_X4eI4MuQNggxYe/view?usp=sharing)  

🔗 [image_captioning_model](https://drive.google.com/file/d/1qvpVw88ChtexlwlzyPsKFds8QP8qWi5R/view?usp=sharing)  

---

### Prerequisites
- Python 3.8+  
- TensorFlow / Keras  
- OpenCV  
- NLTK  

### Steps
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/image-caption-generator.git
   cd image-caption-generator

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download datasets (Tiny ImageNet-200, Flickr30k, Flickr8k) and place them in the `/` directory.

4. Run the model to generate captions:


* **Output:** A generated caption for the input image (e.g., *"A dog is playing in the park."*)

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Contact

📧 **Email:** [umamaheshpalla2004@gmail.com](mailto:umamaheshpalla2004@gmail.com)

⭐ Star the repo if you find it useful!

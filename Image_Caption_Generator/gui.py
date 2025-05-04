import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model, pad_sequences, to_categorical
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, LSTM, add
import os
import pickle
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# clean the caption
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

def extract_features(model, img_path, target_size):
    
    # Load the image with the specified target size
    image = load_img(img_path, target_size=(target_size, target_size))
    
    # Convert image pixels to a numpy array
    image = img_to_array(image)
    image = image.astype('float32')
    
    # Reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    # Preprocess the image (e.g., for VGG)
    image = preprocess_input(image)
    
    # Extract features using the model
    feature = model.predict(image, verbose=0)

    return feature

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# generate caption for an image
def predict_caption(model, feature_vector, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length, padding='post')
        # predict next word
        yhat = model.predict([feature_vector, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
    return in_text

def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('/')[4]
    image_id = image_id.split('.')[0]
    image = Image.open(image_name)
    #captions = mapping[image_id]
    #st.write('---------------------Actual---------------------')
    #for caption in captions:
    #    st.write(caption)

    # get the features
    pre_trained_features = extract_features(pre_trained_model, image_name, 224)
    self_trained_features = extract_features(self_trained_model, image_name, 64)

    # transform the self trained feature vector into a 4096-dimensional vector
    transformation_layer = Dense(4096, activation='relu')
    adjusted_features = transformation_layer(self_trained_features)
        
    # predict the caption
    pre_y_pred = predict_caption(caption_generator_model, pre_trained_features, tokenizer, max_length)
    self_y_pred = predict_caption(caption_generator_model, adjusted_features, tokenizer, max_length)
    
    st.write('--------------------Predicted--------------------')
    st.write("Pre trained model caption:", pre_y_pred)
    st.write("Self trained model caption:", self_y_pred)
    st.image(image)


if __name__ == '__main__':
# get uploaded image and pass it to model
    image = st.file_uploader("Upload an image", accept_multiple_files=False)
    if image is not None:
        MODEL_DIR = '../models/'

        pre_trained_model = load_model("../models/vgg-16.h5")
        self_trained_model = load_model("../models/vgg16_tiny_imagenet_final.h5")
        caption_generator_model = load_model("../models/epo150.keras")
        #caption_generator_model = load_model('/mnt/caption_models/flickr30k_aftermess.h5')

        with open(os.path.join('../dataset/flickr8k/captions.txt'), 'r') as f:
            next(f)
            captions_doc = f.read()

        # create mapping of image to captions
        mapping = {}
        # process lines
        for line in tqdm(captions_doc.split('\n')):
            # split the line by comma(,)
            tokens = line.split(',')
            if len(line) < 2:
                continue
            image_id, caption = tokens[0], tokens[1:]
            # remove extension from image ID
            image_id = image_id.split('.')[0]
            # convert caption list to string
            caption = " ".join(caption)
            # create list if needed
            if image_id not in mapping:
                mapping[image_id] = []
            # store the caption
            mapping[image_id].append(caption)

        # preprocess the text
        clean(mapping)

        # store all the captions
        all_captions = []
        for key in mapping:
            for caption in mapping[key]:
                all_captions.append(caption)

        # tokenize the text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)
        vocab_size = len(tokenizer.word_index) + 1
        max_length = max(len(caption.split()) for caption in all_captions)

        # temp variable to store folder location
        folder_loc = "../dataset/flickr8k/Images/"
        generate_caption(folder_loc + image.name)

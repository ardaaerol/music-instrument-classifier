# Musical Instrument Image Classifier

This project is a computer vision application that classifies images of musical instruments using a ResNet18 model trained with FastAI.

## Dataset

The dataset includes 10 musical instrument categories:
`accordion`, `banjo`, `drum`, `flute`, `guitar`, `harmonica`, `saxophone`, `sitar`, `tabla`, `violin`.

Images were collected from [this Kaggle dataset](https://www.kaggle.com/datasets/nikolasgegenava/music-instruments).

## Training

Training is done in FastAI using the ResNet18 backbone.

- `DataBlock` is used with appropriate `get_items` and `get_y`
- Images are resized and augmented
- Model is fine-tuned with `fine_tune(3)`
- Learning rate was found using `learn.lr_find()`

The model is saved as both `.pkl` and `.pth`.

Training process is documented step-by-step in [`music_instrument_training_full.ipynb`](./music_instrument_training_full.ipynb)

## App

The project is deployed using Gradio and Hugging Face Spaces.

- If model confidence is below 30%, a friendly error is returned
- Otherwise, top predictions are shown

Live demo:  
👉 https://ardaerol28-musicinstrumentclassifier.hf.space

## File Overview

- `app.py`: Main application logic for Gradio
- `requirements.txt`: Python package requirements
- `models/music_weights.pth`: Trained model weights
- `music_instrument_training_full.ipynb`: Training notebook
- `music_instruments_model.pkl`: Exported FastAI model
- Class folders (e.g., `guitar/`, `drum/`): Sample data

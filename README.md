# Real-Time Face Detection Using DLIA Models
This project demonstrates real-time face detection using deep learning models (DLIA). The primary goal is to recognize facial emotions such as angry, happy, and sad using a trained neural network model. The project leverages TensorFlow and Keras for deep learning, along with various libraries such as OpenCV, scikit-learn, and Pandas to manage the image data and perform analysis. The face detection and emotion classification are processed in real-time using a webcam feed.

## Features
* Real-time face detection using OpenCV.
* Facial emotion recognition based on pre-trained deep learning models.
* The ability to classify emotions as angry, happy, and sad.
* Use of grayscale images for simplified and faster processing.
* Easy-to-use implementation via Jupyter Notebook.
## Requirements
### The following Python libraries are required to run this project:

* tensorflow – A powerful open-source machine learning library.
* keras – High-level neural networks API, running on top of TensorFlow.
* pandas – For data manipulation and analysis.
* numpy – For numerical operations and handling multidimensional arrays.
* jupyter – Web-based interactive computing environment for running the notebook.
* notebook – Interactive notebook server for Jupyter.
* tqdm – A fast, extensible progress bar for loops and tasks.
* opencv-contrib-python – For computer vision tasks, including real-time face detection.
* scikit-learn – For data preprocessing, training, and testing machine learning models.
## Installing Dependencies
### To install the required libraries, use the following pip commands:

bash
Copy
pip install tensorflow keras pandas numpy jupyter notebook tqdm opencv-contrib-python scikit-learn
Ensure that you have a Python environment set up (preferably Python 3.x).

## Dataset Structure
The dataset used in this project consists of images of faces categorized by their corresponding emotion. The images are organized into subfolders where each folder represents a specific emotion.

- Copy
images/
  ├── angry/
  │    ├── happy.png
  │    └── h.png
  ├── sad/
  │    ├── happy.png
  │    └── h.png
- images/ – The root directory that contains emotion subfolders.
- angry/ – Folder containing images labeled as angry.
- sad/ – Folder containing images labeled as sad.
- Each image is associated with a label that corresponds to the emotion expressed in the image.
- Example of Image Files
- Image Path	Label
- images/angry/happy.png	angry
- images/angry/h.png	angry
- images/sad/happy.png	sad
- images/sad/h.png	sad
* Each image file contains facial expressions which the model uses to classify the emotion into one of the categories: angry, happy, or sad.

## Code Overview
## 1. Image Preprocessing
We use Keras' load_img function to load and preprocess the images. The images are loaded in grayscale format to simplify the input to the model:

python
Copy
from keras.preprocessing.image import load_img

img = load_img('path_to_image.png', grayscale=True)
This converts the image to grayscale, reducing the complexity of the model and speeding up processing.

## 2. Model Training
The deep learning model is built using Keras and TensorFlow. It is trained on the pre-processed dataset of facial images, where each image has an associated emotion label (angry, happy, or sad). The model is trained to classify the emotion based on facial expressions.

## 3. Real-Time Detection
Using OpenCV, this project enables real-time face detection through the webcam. Once a face is detected, the trained deep learning model predicts the emotion on the detected face. This process is done in real-time, allowing the system to detect emotions as they appear on the screen.

## 4. Libraries Used
- TensorFlow & Keras: For building and training the deep learning model.
- Pandas: For manipulating and analyzing the image dataset.
- NumPy: For performing numerical computations such as image transformations.
- OpenCV: For real-time face detection using a webcam.
- Scikit-learn: For pre-processing the data, splitting into training/testing sets, and using additional machine learning utilities.
TQDM: For visualizing the training process with a progress bar.
## How to Use
#### Step 1: Clone the Repository
Start by cloning the repository to your local machine:

- bash
- Copy
- git clone https://github.com/yourusername/realtime-face-detection.git
- cd realtime-face-detection
#### Step 2: Start Jupyter Notebook
Launch Jupyter Notebook to interact with the code:

- bash
- Copy
- jupyter notebook
- This will open a web interface where you can access the notebook files.

#### Step 3: Run the Notebook
Open the notebook (usually face_detection.ipynb) and follow the steps outlined to train the model or run real-time face detection using your webcam. You can experiment with different facial expressions in front of the camera, and the system will classify them accordingly.

#### Step 4: Model Training (Optional)
If you wish to retrain the model with a different dataset or update the model, simply follow the steps in the notebook to preprocess the data, define the model architecture, and train the model using the labeled images.

#### Step 5: Real-Time Emotion Detection
Once the model is trained, run the real-time face detection system, which will open a webcam feed and classify emotions (angry, happy, sad) based on the faces detected.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements
- Special thanks to the authors of Keras, TensorFlow, OpenCV, and scikit-learn for providing powerful tools that helped make this project possible.
- The dataset used for training the model is sourced from publicly available facial emotion datasets.
- TQDM was used for displaying progress bars during training and other long-running tasks.
- Troubleshooting
- If your webcam is not detected, ensure that your OpenCV installation is correctly set up. You may need to install or update additional camera drivers.
- If the model is not performing well, consider retraining with a larger or more diverse dataset.
- If you encounter any errors or issues, feel free to open an issue in the repository, and we will assist you as soon as possible.


  ### * The dataset is available from this link = https://drive.google.com/drive/folders/1S3I21Cz7flLxwxypHwFCb5JeRSirGLMZ

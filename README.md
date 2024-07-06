
# 1.  Deepfake Detection Model
This repository contains a deep learning model for detecting deepfake images. The model is built using TensorFlow and Keras, and it employs a Convolutional Neural Network (CNN) to classify images as either "original" or "deepfake". 

## 2. Table of Contents
=======
# Deepfake Detection Model

This repository contains a deep learning model for detecting deepfake images. The model is built using TensorFlow and Keras, and it employs a Convolutional Neural Network (CNN) to classify images as either "original" or "deepfake". 

## Table of Contents
>>>>>>> dcfc5e29954f6704b0d5d8a2ff76ce276d99835d

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Saving and Loading Model](#saving-and-loading-model)
- [Contributing](#contributing)
<<<<<<< HEAD


## 3. Installation
=======
- [License](#license)

## Installation
>>>>>>> dcfc5e29954f6704b0d5d8a2ff76ce276d99835d

To run the code in this repository, you need to have Python and the following libraries installed:

- TensorFlow
- NumPy
- Matplotlib
- OpenCV

You can install the required packages using pip:

```bash
pip install tensorflow numpy matplotlib opencv-python
<<<<<<< HEAD
```

## 4. Usage
Clone the repository:
```bash
git clone https://github.com/Auraoflegend/deepfake.git

```
## 5. Model Architecture
The model is a Convolutional Neural Network (CNN) with the following layers:

- Three convolutional layers with ReLU activation and max-pooling
- A flattening layer

- Two dense layers, with the final layer using a sigmoid activation function

## 6. Training
The model is trained on images from the train directory. The training data is split into training, validation, and test sets. The images are normalized by scaling the pixel values to the range [0, 1]. Data augmentation is applied to increase the diversity of the training set.

## 7. Evaluation
After training, the model is evaluated on the test set using precision, recall, and accuracy metrics.

```bash
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f'Precision: {pre.result().numpy() * 100}, Recall: {re.result().numpy() * 100}, Accuracy: {acc.result().numpy() * 100}')

```

## 8. Testing
To test the model on a single image, load the image using OpenCV, resize it, and make a prediction.



Copy code

```bash import cv2
img = cv2.imread('test_file_path')
resize = tf.image.resize(img, (256, 256))
yhat = model.predict(np.expand_dims(resize / 255, 0))

if yhat > 0.5:
    print(f'Predicted class is original')
else:
    print(f'Predicted class is deepfake')

```

## 9. Saving and Loading Model
The trained model can be saved to a file and later loaded for making predictions.

```bash
model.save(os.path.join('models', 'deepfakemodel.h5'))
new_model = tf.keras.models.load_model(os.path.join('models', 'deepfakemodel.h5'))
yhatnew = new_model.predict(np.expand_dims(resize / 255, 0))

```

## 10. Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss changes.

## Thankyou


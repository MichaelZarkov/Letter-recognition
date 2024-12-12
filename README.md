## Intro

This project showcases the classical machine learning problem of classifying handwritten characters.
The final result is a Python script (and a standalone Windows program) that lets the user draw a 
letter on the screen (english or bulgarian), and then a pre-trained convolutional neural network
(CNN) tries to guess what it is.

## Model Training

The notebook files `train-en-model.ipynb` and `train-bg-model.ipynb` deal with the preprocessing
of the datasets, definition and training of the english and bulgarian models. The two models are
almost identical:
- input is 28x28 grayscale image
- same hidden layers on both (except for the dropout layer)
- output - 26 dimensional vector for the english model (A-Z); 30 dim. vector for bulgarian (А-Я)

After training accuracy on the testing sets was:
- ~ 0.943 - english
- ~ 0.975 - bulgarian

_Tensorflow_ Python library was used for model creation and training (mainly for its ease of use),
then the models were converted from _Keras_ to _ONNX_ open source format. _ONNX Runtime_ was used
for the final demo program (for faster startup and smaller program size compared to Tensorflow).

### Datasets

[EMNIST letters][emnist-letters] dataset was used for training the English model, consisting of
145,600 images of handwritten English letters. 

[Cyrillic-MNIST][cyrillic-mnist] dataset was used for the Bulgarian model consisting of 77,443
images of handwritten Bulgarian letters. Some Cyrillic letters which are not part of the Bulgarian
alphabet were excluded. Dataset is available on [GitHub][cyrillic-mnist-github].

## Final Program

The final program is in the `app` folder. **You don't need to install Python to run it**. Just 
download the contents of `app` and run `draw-letters.exe`. Instructions on how to use it are 
provided in the corresponding `README.md` (in `app`).

Alternatively if you have Python installed together with the necessary modules (_opencv-python_, 
_numpy_, _onnxruntime_), you can download the folders `imgs`, `models`, `srs` and run the script
`draw-letters.py`.

[cyrillic-mnist]: https://aclanthology.org/2022.lrec-1.510/
[cyrillic-mnist-github]: https://github.com/bolattleubayev/cmnist
[emnist-letters]: https://www.nist.gov/itl/products-and-services/emnist-dataset
import cv2
import numpy as np
import onnxruntime as ort
import sys
import os

class Callback:
    def __init__(self):
        self.is_drawing = False
        self.point_x, self.point_y = None, None  # Coordinates of point when last mouse event happened.
        self.img = np.zeros((512, 512, 3), np.uint8)  # screen

    def mouse_callback(self, event, x, y, flags, param):
        """Draw with mouse on 'self.img'."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.point_x, self.point_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                cv2.line(self.img, (self.point_x, self.point_y), (x, y), color=(255, 255, 255), thickness=30)
                self.point_x, self.point_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            cv2.line(self.img, (self.point_x, self.point_y), (x, y), color=(255, 255, 255), thickness=30)

    def reset_img(self):
        """Resets the image inplace to black."""
        self.img -= self.img

def load_letters(directory):
    """Returns a list of the images of alphabet letters ordered alphabetically."""
    letter_imgs = list()
    img_filenames = os.listdir(directory)
    img_filenames.sort()
    for f in img_filenames:
        letter_imgs.append(cv2.resize(cv2.imread(os.path.join(directory,f)), (512,512)))
    return letter_imgs

def process_image(image, negative=False):
    """Returns an image suitable to feed into the neural network."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    if negative:
        image = cv2.bitwise_not(image)
    image = image.astype(np.float32) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

def main():
    """Simple script to showcase the English and Bulgarian convolutional neural network models.
    When you start the script a black window will pop up. You can draw a letter by holding the
    left mouse button and moving the mouse. Then you can press '1' on the keyboard to make a prediction
    with the english model, or press '2' to make a prediction with the bulgarian model. After that,
    press any button to clear the prediction.
    You exit the app by pressing 'esc'."""

    # Set up a window.
    cb = Callback()
    cv2.namedWindow('Draw letter')
    cv2.setMouseCallback('Draw letter', cb.mouse_callback)

    # Load models.
    # 'onnxruntime' is used instead of 'tensorflow' because onnxruntime is lightweight compared to tensorflow.
    en_model = ort.InferenceSession('../models/en-model.onnx')
    bg_model = ort.InferenceSession('../models/bg-model.onnx')

    en_imgs = load_letters('../imgs/en')
    bg_imgs = load_letters('../imgs/bg')

    while True:
        cv2.imshow('Draw letter', cb.img)
        key_press = cv2.waitKey(1)

        # `esc` - quit.
        if key_press == 27:
            cv2.destroyAllWindows()
            sys.exit()
        # '1' - use english model.
        elif key_press == ord('1'):
            image = process_image(cb.img)
            predicted_class = np.argmax(en_model.run(output_names=None, input_feed={'input': image}))
            cv2.imshow('Draw letter', en_imgs[predicted_class])
            cv2.waitKey(0)
            cb.reset_img()
        # '2' - use cyrillic model.
        elif key_press == ord('2'):
            image = process_image(cb.img, negative=True)  # Model is trained on black letters on white background.
            predicted_class = np.argmax(bg_model.run(output_names=None, input_feed={'input': image}))
            cv2.imshow('Draw letter', bg_imgs[predicted_class])
            cv2.waitKey(0)
            cb.reset_img()

if __name__ == "__main__":
    main()
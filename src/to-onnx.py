import onnx
import tf2onnx
import tensorflow as tf

def keras_to_onnx():
    """Convert the keras models to onnx models."""

    en_model = tf.keras.models.load_model('../models/en-model.keras', compile=False)
    bg_model = tf.keras.models.load_model('../models/bg-model.keras', compile=False)

    # See the links below why the following three lines of code are necessary.
    # https://onnxruntime.ai/docs/tutorials/tf-get-started.html#keras-models-and-tf-functions
    # https://github.com/onnx/tensorflow-onnx/issues/2319#issuecomment-2522389819
    input_signature = [tf.TensorSpec(en_model.inputs[0].shape, en_model.inputs[0].dtype, name='input')]
    en_model.output_names = ['output']
    bg_model.output_names = ['output']

    en_onnx_model, _ = tf2onnx.convert.from_keras(en_model, input_signature)
    bg_onnx_model, _ = tf2onnx.convert.from_keras(bg_model, input_signature)

    onnx.save(en_onnx_model, '../models/en-model.onnx')
    onnx.save(bg_onnx_model, '../models/bg-model.onnx')

if __name__ == '__main__':
    keras_to_onnx()
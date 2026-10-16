"""
Builds TWO files:
  app/src/main/assets/feature_extractor.tflite  — frozen MobileNetV2, ~9-14MB
  app/src/main/assets/head_model.tflite          — tiny trainable head, ~20-400KB

Run this ONCE on a regular machine (not on-device) with TensorFlow installed:
    pip install tensorflow==2.16.1
    python tools/build_base_model.py
"""

import tensorflow as tf

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
BOTTLENECK_SIZE = 1280
NUM_CLASSES = 4
LEARNING_RATE = 0.001


def build_feature_extractor():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False
    return base


class TrainableHead(tf.Module):
    def __init__(self):
        super().__init__()
        initializer = tf.keras.initializers.GlorotUniform()
        self.w = tf.Variable(
            initializer([BOTTLENECK_SIZE, NUM_CLASSES]), trainable=True, name="head/kernel"
        )
        self.b = tf.Variable(tf.zeros([NUM_CLASSES]), trainable=True, name="head/bias")
        self._trainable_vars = [self.w, self.b]
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def _forward(self, bottleneck):
        logits = tf.matmul(bottleneck, self.w) + self.b
        return tf.nn.softmax(logits)

    @tf.function(input_signature=[
        tf.TensorSpec([None, BOTTLENECK_SIZE], tf.float32, name="bottleneck"),
        tf.TensorSpec([None, NUM_CLASSES], tf.float32, name="labels"),
    ])
    def train(self, bottleneck, labels):
        with tf.GradientTape() as tape:
            preds = self._forward(bottleneck)
            loss = self.loss_fn(labels, preds)
        grads = tape.gradient(loss, self._trainable_vars)
        self.optimizer.apply_gradients(zip(grads, self._trainable_vars))
        return {"loss": loss}

    @tf.function(input_signature=[
        tf.TensorSpec([None, BOTTLENECK_SIZE], tf.float32, name="bottleneck"),
    ])
    def infer(self, bottleneck):
        return {"logits": self._forward(bottleneck)}

    @tf.function(input_signature=[tf.TensorSpec([], tf.string, name="checkpoint_path")])
    def save(self, checkpoint_path):
        tensor_names = [v.name for v in self._trainable_vars]
        tensors_to_save = [tf.convert_to_tensor(v) for v in self._trainable_vars]
        tf.raw_ops.Save(
            filename=checkpoint_path,
            tensor_names=tensor_names,
            data=tensors_to_save,
            name="save",
        )
        return {"checkpoint_path": checkpoint_path}

    @tf.function(input_signature=[tf.TensorSpec([], tf.string, name="checkpoint_path")])
    def restore(self, checkpoint_path):
        restored = {}
        for v in self._trainable_vars:
            restored_tensor = tf.raw_ops.Restore(
                file_pattern=checkpoint_path,
                tensor_name=v.name,
                dt=v.dtype,
                name="restore",
            )
            restored_tensor.set_shape(v.shape)
            v.assign(restored_tensor)
            restored[v.name] = restored_tensor
        return restored


def export_feature_extractor():
    base = build_feature_extractor()
    converter = tf.lite.TFLiteConverter.from_keras_model(base)
    tflite_model = converter.convert()
    out_path = "app/src/main/assets/feature_extractor.tflite"
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"wrote {out_path} ({len(tflite_model)} bytes)")


def export_head_model():
    head = TrainableHead()
    saved_model_dir = "tools/_head_saved_model"
    tf.saved_model.save(
        head,
        saved_model_dir,
        signatures={
            "train": head.train,
            "infer": head.infer,
            "save": head.save,
            "restore": head.restore,
        },
    )
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()
    out_path = "app/src/main/assets/head_model.tflite"
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"wrote {out_path} ({len(tflite_model)} bytes)")


def main():
    export_feature_extractor()
    export_head_model()


if __name__ == "__main__":
    main()

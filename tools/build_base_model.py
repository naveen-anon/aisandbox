"""
Builds app/src/main/assets/base_model.tflite

Run this ONCE on a regular machine (not on-device) with TensorFlow installed:
    pip install tensorflow==2.16.1
    python tools/build_base_model.py

What it produces: a TFLite graph exported with four *signatures* —
train / infer / save / restore — which is Google's supported pattern for
on-device training (see tensorflow/examples: lite/examples/model_personalization).

Architecture: a frozen MobileNetV2 feature extractor (ImageNet weights,
not trainable) + a small trainable Dense head. Only the head's weights
get updated by SandboxTrainer.train() inside the isolated process — this
keeps on-device training fast enough to actually run on a phone CPU.

If you change NUM_CLASSES / IMAGE_WIDTH / IMAGE_HEIGHT, update
ModelSpec.kt (app/src/main/java/com/naveen/aisandbox/ipc/TrainingProtocol.kt)
to match, or the Kotlin side will build tensors of the wrong shape.
"""

import tensorflow as tf

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
NUM_CLASSES = 4
LEARNING_RATE = 0.001


def build_keras_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False  # frozen — never updated on-device

    inputs = tf.keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), name="images")
    x = base(inputs, training=False)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="head")(x)
    return tf.keras.Model(inputs, outputs), base


class TrainableModule(tf.Module):
    """Wraps the Keras model with the 4 signatures TFLite needs for on-device training."""

    def __init__(self, model: tf.keras.Model):
        super().__init__()
        self.model = model
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        # only the Dense head has trainable variables, since the backbone is frozen
        self._trainable_vars = model.trainable_variables

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], tf.float32, name="images"),
        tf.TensorSpec([None, NUM_CLASSES], tf.float32, name="labels"),
    ])
    def train(self, images, labels):
        with tf.GradientTape() as tape:
            preds = self.model(images, training=True)
            loss = self.loss_fn(labels, preds)
        grads = tape.gradient(loss, self._trainable_vars)
        self.optimizer.apply_gradients(zip(grads, self._trainable_vars))
        return {"loss": loss}

    @tf.function(input_signature=[
        tf.TensorSpec([None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], tf.float32, name="images"),
    ])
    def infer(self, images):
        return {"logits": self.model(images, training=False)}

    @tf.function(input_signature=[tf.TensorSpec([], tf.string, name="checkpoint_path")])
    def save(self, checkpoint_path):
        tensor_names = [v.name for v in self._trainable_vars]
        tensors_to_save = [v.read_value() for v in self._trainable_vars]
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
            v.assign(restored_tensor)
            restored[v.name] = restored_tensor
        return restored


def main():
    keras_model, _ = build_keras_model()
    module = TrainableModule(keras_model)

    saved_model_dir = "tools/_saved_model"
    tf.saved_model.save(
        module,
        saved_model_dir,
        signatures={
            "train": module.train,
            "infer": module.infer,
            "save": module.save,
            "restore": module.restore,
        },
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,  # Save/Restore raw ops need this
    ]
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    out_path = "app/src/main/assets/base_model.tflite"
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"wrote {out_path} ({len(tflite_model)} bytes)")


if __name__ == "__main__":
    main()

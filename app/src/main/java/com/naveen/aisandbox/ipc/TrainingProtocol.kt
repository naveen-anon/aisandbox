package com.naveen.aisandbox.ipc

object TrainingProtocol {
    const val MSG_INIT_MODEL = 1
    const val MSG_ADD_SAMPLE = 2
    const val MSG_START_TRAINING = 3
    const val MSG_CANCEL_TRAINING = 4

    const val MSG_TRAINING_PROGRESS = 100
    const val MSG_TRAINING_COMPLETE = 101
    const val MSG_TRAINING_ERROR = 102
    const val MSG_READY = 103

    const val KEY_FEATURE_MODEL_BYTES = "feature_model_bytes"
    const val KEY_HEAD_MODEL_BYTES = "head_model_bytes"
    const val KEY_IMAGE_BYTES = "image_bytes"
    const val KEY_LABEL_INDEX = "label_index"
    const val KEY_EPOCHS = "epochs"
    const val KEY_BATCH_SIZE = "batch_size"
    const val KEY_LEARNING_RATE = "learning_rate"
    const val KEY_EPOCH = "epoch"
    const val KEY_STEP = "step"
    const val KEY_LOSS = "loss"
    const val KEY_CHECKPOINT_BYTES = "checkpoint_bytes"
    const val KEY_FINAL_LOSS = "final_loss"
    const val KEY_ERROR_MESSAGE = "error_message"
}

object ModelSpec {
    const val IMAGE_WIDTH = 224
    const val IMAGE_HEIGHT = 224
    const val IMAGE_CHANNELS = 3
    const val BOTTLENECK_SIZE = 1280
    const val NUM_CLASSES = 4
    const val SIG_TRAIN = "train"
    const val SIG_INFER = "infer"
    const val SIG_SAVE = "save"
    const val SIG_RESTORE = "restore"
}

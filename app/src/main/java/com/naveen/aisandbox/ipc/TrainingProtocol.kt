package com.naveen.aisandbox.ipc

/**
 * Wire protocol between the main (UI) process and the isolated
 * `:training_sandbox` process. Nothing crosses this boundary except these
 * message types — the sandbox never gets a Context tied to the real app,
 * a file path, or a ContentResolver. Everything is raw bytes in a Bundle.
 */
object TrainingProtocol {

    // Main process -> sandbox
    const val MSG_INIT_MODEL = 1          // Bundle: KEY_MODEL_BYTES (base tflite graph, read-only asset copy)
    const val MSG_ADD_SAMPLE = 2          // Bundle: KEY_IMAGE_BYTES, KEY_LABEL_INDEX
    const val MSG_START_TRAINING = 3      // Bundle: KEY_EPOCHS, KEY_BATCH_SIZE, KEY_LEARNING_RATE
    const val MSG_CANCEL_TRAINING = 4

    // Sandbox -> main process
    const val MSG_TRAINING_PROGRESS = 100 // Bundle: KEY_EPOCH, KEY_STEP, KEY_LOSS
    const val MSG_TRAINING_COMPLETE = 101 // Bundle: KEY_CHECKPOINT_BYTES, KEY_FINAL_LOSS
    const val MSG_TRAINING_ERROR = 102    // Bundle: KEY_ERROR_MESSAGE
    const val MSG_READY = 103             // sandbox finished MSG_INIT_MODEL successfully

    const val KEY_MODEL_BYTES = "model_bytes"
    const val KEY_IMAGE_BYTES = "image_bytes"      // raw RGB bytes, fixed size, see ModelSpec
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

/** Must match the signature the base .tflite graph was exported with (see tools/build_base_model.py). */
object ModelSpec {
    const val IMAGE_WIDTH = 224
    const val IMAGE_HEIGHT = 224
    const val IMAGE_CHANNELS = 3
    const val NUM_CLASSES = 4          // change to match your labels, and re-export the base model
    const val SIG_TRAIN = "train"
    const val SIG_INFER = "infer"
    const val SIG_SAVE = "save"
    const val SIG_RESTORE = "restore"
}

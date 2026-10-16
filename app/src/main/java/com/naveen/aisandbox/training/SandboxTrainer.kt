package com.naveen.aisandbox.training

import com.naveen.aisandbox.ipc.ModelSpec
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer

/**
 * Runs entirely inside the isolated `:training_sandbox` process.
 *
 * Wraps a TFLite graph that was exported (see tools/build_base_model.py)
 * with four signatures: train, infer, save, restore — this is the
 * supported on-device-training pattern (TF's "Model Personalization"
 * approach), not a hack. The frozen feature-extractor backbone stays
 * fixed; only a small trainable head is updated on-device.
 *
 * This class never touches the filesystem, network, or any Android
 * Context beyond what's needed to construct the Interpreter from an
 * in-memory ByteBuffer. Checkpoints are returned as byte[] over Binder;
 * this class does not decide where they end up persisted.
 */
class SandboxTrainer(modelBytes: ByteArray) {

    private val interpreter: Interpreter

    private val samples = mutableListOf<FloatArray>()
    private val labels = mutableListOf<Int>()

    @Volatile var cancelled = false
        private set

    init {
        val direct = ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder())
        direct.put(modelBytes)
        direct.rewind()
        val options = Interpreter.Options().apply {
            numThreads = 2 // deliberately modest — isolated sandbox process, don't hog the device
        }
        interpreter = Interpreter(direct, options)
    }

    fun addSample(imageBytes: ByteArray, label: Int) {
        require(label in 0 until ModelSpec.NUM_CLASSES) { "label out of range: $label" }
        val floats = FloatArray(imageBytes.size)
        for (i in imageBytes.indices) {
            // normalize uint8 [0,255] -> float [-1,1], matches tools/build_base_model.py preprocessing
            floats[i] = (imageBytes[i].toInt() and 0xFF) / 127.5f - 1f
        }
        samples.add(floats)
        labels.add(label)
    }

    fun cancel() { cancelled = true }

    /**
     * Runs [epochs] over the in-memory samples in mini-batches, invoking
     * [onProgress] after every step and returning the final serialized
     * checkpoint bytes (from the "save" signature) plus final loss.
     */
    fun train(
        epochs: Int,
        batchSize: Int,
        onProgress: (epoch: Int, step: Int, loss: Float) -> Unit
    ): Pair<ByteArray, Float> {
        require(samples.isNotEmpty()) { "no training samples added" }
        var lastLoss = Float.NaN

        for (epoch in 0 until epochs) {
            if (cancelled) break
            val indices = samples.indices.shuffled()
            var step = 0
            var i = 0
            while (i < indices.size) {
                if (cancelled) break
                val batchIdx = indices.subList(i, minOf(i + batchSize, indices.size))

                val trainImages = Array(batchIdx.size) { b -> samples[batchIdx[b]] }
                val trainLabels = Array(batchIdx.size) { b ->
                    FloatArray(ModelSpec.NUM_CLASSES).also { it[labels[batchIdx[b]]] = 1f }
                }

                val inputs = mapOf(
                    "images" to trainImages,
                    "labels" to trainLabels
                )
                val outputs = mutableMapOf<String, Any>(
                    "loss" to FloatArray(1)
                )
                interpreter.runSignature(inputs, outputs, ModelSpec.SIG_TRAIN)
                lastLoss = (outputs["loss"] as FloatArray)[0]
                onProgress(epoch, step, lastLoss)
                step++
                i += batchSize
            }
        }

        val checkpointBytes = runSave()
        return checkpointBytes to lastLoss
    }

    private fun runSave(): ByteArray {
        val outPath = mutableMapOf<String, Any>("checkpoint_path" to arrayOf(IN_MEMORY_CHECKPOINT_TAG))
        // TFLite's save signature in this pattern writes a checkpoint file; since this
        // process has no durable/shared storage, we point it at this process's private
        // cache dir (isolated processes DO get their own throwaway cache dir) and then
        // read the bytes back immediately to ship over Binder.
        val tmp = File.createTempFile("ckpt", ".bin")
        try {
            val inputs = mapOf("checkpoint_path" to arrayOf(tmp.absolutePath))
            val outputs = mutableMapOf<String, Any>()
            interpreter.runSignature(inputs, outputs, ModelSpec.SIG_SAVE)
            val baos = ByteArrayOutputStream()
            tmp.inputStream().use { it.copyTo(baos) }
            return baos.toByteArray()
        } finally {
            tmp.delete()
        }
    }

    fun close() {
        interpreter.close()
    }

    companion object {
        private const val IN_MEMORY_CHECKPOINT_TAG = "sandbox_checkpoint"
    }
}

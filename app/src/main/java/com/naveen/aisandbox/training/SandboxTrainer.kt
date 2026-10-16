package com.naveen.aisandbox.training

import com.naveen.aisandbox.ipc.ModelSpec
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SandboxTrainer(featureModelBytes: ByteArray, headModelBytes: ByteArray) {

    private val featureInterpreter: Interpreter
    private val headInterpreter: Interpreter

    private val bottlenecks = mutableListOf<FloatArray>()
    private val labels = mutableListOf<Int>()

    @Volatile var cancelled = false
        private set

    init {
        val opts = Interpreter.Options().apply { numThreads = 2 }
        featureInterpreter = Interpreter(toDirectBuffer(featureModelBytes), opts)
        headInterpreter = Interpreter(toDirectBuffer(headModelBytes), opts)
    }

    private fun toDirectBuffer(bytes: ByteArray): ByteBuffer =
        ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply {
            put(bytes)
            rewind()
        }

    fun addSample(imageBytes: ByteArray, label: Int) {
        require(label in 0 until ModelSpec.NUM_CLASSES) { "label out of range: $label" }
        val input = Array(1) {
            Array(ModelSpec.IMAGE_HEIGHT) {
                Array(ModelSpec.IMAGE_WIDTH) { FloatArray(ModelSpec.IMAGE_CHANNELS) }
            }
        }
        var idx = 0
        for (y in 0 until ModelSpec.IMAGE_HEIGHT) {
            for (x in 0 until ModelSpec.IMAGE_WIDTH) {
                for (c in 0 until ModelSpec.IMAGE_CHANNELS) {
                    val v = (imageBytes[idx].toInt() and 0xFF) / 127.5f - 1f
                    input[0][y][x][c] = v
                    idx++
                }
            }
        }
        val output = Array(1) { FloatArray(ModelSpec.BOTTLENECK_SIZE) }
        featureInterpreter.run(input, output)
        bottlenecks.add(output[0])
        labels.add(label)
    }

    fun cancel() { cancelled = true }

    fun train(
        epochs: Int,
        batchSize: Int,
        onProgress: (epoch: Int, step: Int, loss: Float) -> Unit
    ): Pair<ByteArray, Float> {
        require(bottlenecks.isNotEmpty()) { "no training samples added" }
        var lastLoss = Float.NaN

        for (epoch in 0 until epochs) {
            if (cancelled) break
            val indices = bottlenecks.indices.shuffled()
            var step = 0
            var i = 0
            while (i < indices.size) {
                if (cancelled) break
                val batchIdx = indices.subList(i, minOf(i + batchSize, indices.size))

                val batchBottlenecks = Array(batchIdx.size) { b -> bottlenecks[batchIdx[b]] }
                val batchLabels = Array(batchIdx.size) { b ->
                    FloatArray(ModelSpec.NUM_CLASSES).also { it[labels[batchIdx[b]]] = 1f }
                }

                val inputs = mapOf(
                    "bottleneck" to batchBottlenecks,
                    "labels" to batchLabels
                )
                val outputs = mutableMapOf<String, Any>("loss" to FloatArray(1))
                headInterpreter.runSignature(inputs, outputs, ModelSpec.SIG_TRAIN)
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
        val tmp = File.createTempFile("ckpt", ".bin")
        try {
            val inputs = mapOf("checkpoint_path" to arrayOf(tmp.absolutePath))
            val outputs = mutableMapOf<String, Any>()
            headInterpreter.runSignature(inputs, outputs, ModelSpec.SIG_SAVE)
            val baos = ByteArrayOutputStream()
            tmp.inputStream().use { it.copyTo(baos) }
            return baos.toByteArray()
        } finally {
            tmp.delete()
        }
    }

    fun close() {
        featureInterpreter.close()
        headInterpreter.close()
    }
}

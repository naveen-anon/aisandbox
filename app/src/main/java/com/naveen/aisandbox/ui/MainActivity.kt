package com.naveen.aisandbox.ui

import android.graphics.Bitmap
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.naveen.aisandbox.data.CheckpointStore
import com.naveen.aisandbox.databinding.ActivityMainBinding
import com.naveen.aisandbox.ipc.ModelSpec
import com.naveen.aisandbox.training.SandboxController
import java.nio.ByteBuffer
import java.util.UUID

class MainActivity : AppCompatActivity(), SandboxController.Listener {

    private lateinit var binding: ActivityMainBinding
    private lateinit var controller: SandboxController
    private lateinit var checkpoints: CheckpointStore

    private var labeledCount = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        checkpoints = CheckpointStore(this)
        controller = SandboxController(this).also { it.listener = this }

        binding.btnStartSandbox.setOnClickListener {
            controller.bind()
            setStatus("Binding isolated training process…")
        }

        binding.btnLoadModel.setOnClickListener {
            try {
                val featureAfd = assets.openFd("feature_extractor.tflite")
                val headAfd = assets.openFd("head_model.tflite")
                controller.initModel(featureAfd, headAfd)
                setStatus("Sent model FDs to sandbox (feature=${featureAfd.length}B, head=${headAfd.length}B)")
            } catch (e: Exception) {
                setStatus("ERROR: feature_extractor.tflite / head_model.tflite missing or not openFd-able — " +
                    "run tools/build_base_model.py, commit output, rebuild. " +
                    "(${e.javaClass.simpleName}: ${e.message})")
            }
        }

        binding.btnAddSample.setOnClickListener {
            val label = binding.spinnerLabel.selectedItemPosition
            val dummyImage = ByteArray(ModelSpec.IMAGE_WIDTH * ModelSpec.IMAGE_HEIGHT * ModelSpec.IMAGE_CHANNELS)
            controller.addSample(dummyImage, label)
            labeledCount++
            setStatus("Samples queued: $labeledCount")
        }

        binding.btnTrain.setOnClickListener {
            controller.startTraining(epochs = 10, batchSize = 8)
            setStatus("Training started in isolated process…")
        }

        binding.btnCancel.setOnClickListener {
            controller.cancelTraining()
        }
    }

    override fun onDestroy() {
        controller.unbind()
        super.onDestroy()
    }

    override fun onReady() = setStatus("Sandbox ready — add samples, then Train")

    override fun onProgress(epoch: Int, step: Int, loss: Float) =
        setStatus("epoch $epoch step $step  loss=${"%.4f".format(loss)}")

    override fun onComplete(checkpoint: ByteArray, finalLoss: Float) {
        val name = "model_${UUID.randomUUID().toString().take(8)}"
        val f = checkpoints.save(name, checkpoint)
        setStatus("Done. loss=${"%.4f".format(finalLoss)} saved -> ${f.name} (${checkpoint.size} bytes)")
    }

    override fun onError(message: String) = setStatus("ERROR: $message")

    private fun setStatus(s: String) {
        binding.tvStatus.text = s
    }
}

package com.naveen.aisandbox.training

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.*
import com.naveen.aisandbox.ipc.TrainingProtocol

/**
 * Runs in the normal app process. Owns the bound connection to the
 * isolated `:training_sandbox` process and translates its Messenger
 * protocol into plain Kotlin callbacks for the UI layer.
 *
 * This is the ONLY class in the main process allowed to talk to the
 * sandbox. Keep it that way — don't let Activities bind directly.
 */
class SandboxController(private val context: Context) {

    interface Listener {
        fun onReady() {}
        fun onProgress(epoch: Int, step: Int, loss: Float) {}
        fun onComplete(checkpoint: ByteArray, finalLoss: Float) {}
        fun onError(message: String) {}
    }

    var listener: Listener? = null

    private var serviceMessenger: Messenger? = null
    private var bound = false

    private val clientMessenger = Messenger(Handler(Looper.getMainLooper()) { msg ->
        val data = msg.data ?: Bundle()
        when (msg.what) {
            TrainingProtocol.MSG_READY -> listener?.onReady()
            TrainingProtocol.MSG_TRAINING_PROGRESS -> listener?.onProgress(
                data.getInt(TrainingProtocol.KEY_EPOCH),
                data.getInt(TrainingProtocol.KEY_STEP),
                data.getFloat(TrainingProtocol.KEY_LOSS)
            )
            TrainingProtocol.MSG_TRAINING_COMPLETE -> listener?.onComplete(
                data.getByteArray(TrainingProtocol.KEY_CHECKPOINT_BYTES) ?: ByteArray(0),
                data.getFloat(TrainingProtocol.KEY_FINAL_LOSS)
            )
            TrainingProtocol.MSG_TRAINING_ERROR -> listener?.onError(
                data.getString(TrainingProtocol.KEY_ERROR_MESSAGE) ?: "unknown sandbox error"
            )
        }
        true
    })

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            serviceMessenger = Messenger(binder)
            bound = true
        }
        override fun onServiceDisconnected(name: ComponentName?) {
            serviceMessenger = null
            bound = false
        }
    }

    fun bind() {
        val intent = Intent(context, TrainingService::class.java)
        context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
    }

    fun unbind() {
        if (bound) {
            context.unbindService(connection)
            bound = false
        }
        serviceMessenger = null
    }

    fun initModel(modelBytes: ByteArray) {
        sendToSandbox(TrainingProtocol.MSG_INIT_MODEL, Bundle().apply {
            putByteArray(TrainingProtocol.KEY_MODEL_BYTES, modelBytes)
        })
    }

    fun addSample(imageBytes: ByteArray, labelIndex: Int) {
        sendToSandbox(TrainingProtocol.MSG_ADD_SAMPLE, Bundle().apply {
            putByteArray(TrainingProtocol.KEY_IMAGE_BYTES, imageBytes)
            putInt(TrainingProtocol.KEY_LABEL_INDEX, labelIndex)
        })
    }

    fun startTraining(epochs: Int, batchSize: Int) {
        sendToSandbox(TrainingProtocol.MSG_START_TRAINING, Bundle().apply {
            putInt(TrainingProtocol.KEY_EPOCHS, epochs)
            putInt(TrainingProtocol.KEY_BATCH_SIZE, batchSize)
        })
    }

    fun cancelTraining() {
        sendToSandbox(TrainingProtocol.MSG_CANCEL_TRAINING, Bundle())
    }

    private fun sendToSandbox(what: Int, data: Bundle) {
        val target = serviceMessenger ?: run {
            listener?.onError("sandbox not bound yet")
            return
        }
        val msg = Message.obtain(null, what).apply {
            this.data = data
            replyTo = clientMessenger
        }
        try {
            target.send(msg)
        } catch (e: RemoteException) {
            listener?.onError("sandbox process died: ${e.message}")
        }
    }
}

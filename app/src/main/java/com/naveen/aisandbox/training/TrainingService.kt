package com.naveen.aisandbox.training

import android.app.Service
import android.content.Intent
import android.os.*
import com.naveen.aisandbox.ipc.TrainingProtocol
import kotlin.concurrent.thread

class TrainingService : Service() {

    private var trainer: SandboxTrainer? = null
    private var replyTo: Messenger? = null

    private val incomingHandler = Handler(Looper.getMainLooper()) { msg ->
        try {
            handle(msg)
        } catch (t: Throwable) {
            sendError(t.message ?: t.javaClass.simpleName)
        }
        true
    }
    private val messenger = Messenger(incomingHandler)

    override fun onBind(intent: Intent?): IBinder = messenger.binder

    override fun onDestroy() {
        trainer?.close()
        trainer = null
        super.onDestroy()
    }

    private fun handle(msg: Message) {
        replyTo = msg.replyTo ?: replyTo
        val data = msg.data ?: Bundle()

        when (msg.what) {
            TrainingProtocol.MSG_INIT_MODEL -> {
                data.classLoader = ParcelFileDescriptor::class.java.classLoader
                val featureFd = data.getParcelable<ParcelFileDescriptor>(TrainingProtocol.KEY_FEATURE_MODEL_FD)
                    ?: error("no feature model fd supplied")
                val featureOffset = data.getLong(TrainingProtocol.KEY_FEATURE_MODEL_OFFSET)
                val featureLength = data.getLong(TrainingProtocol.KEY_FEATURE_MODEL_LENGTH)
                val headFd = data.getParcelable<ParcelFileDescriptor>(TrainingProtocol.KEY_HEAD_MODEL_FD)
                    ?: error("no head model fd supplied")
                val headOffset = data.getLong(TrainingProtocol.KEY_HEAD_MODEL_OFFSET)
                val headLength = data.getLong(TrainingProtocol.KEY_HEAD_MODEL_LENGTH)

                trainer?.close()
                trainer = try {
                    SandboxTrainer(featureFd, featureOffset, featureLength, headFd, headOffset, headLength)
                } finally {
                    featureFd.close()
                    headFd.close()
                }
                send(TrainingProtocol.MSG_READY, Bundle())
            }

            TrainingProtocol.MSG_ADD_SAMPLE -> {
                val img = data.getByteArray(TrainingProtocol.KEY_IMAGE_BYTES) ?: error("no image bytes")
                val label = data.getInt(TrainingProtocol.KEY_LABEL_INDEX, -1)
                requireTrainer().addSample(img, label)
            }

            TrainingProtocol.MSG_START_TRAINING -> {
                val epochs = data.getInt(TrainingProtocol.KEY_EPOCHS, 5)
                val batchSize = data.getInt(TrainingProtocol.KEY_BATCH_SIZE, 8)
                val t = requireTrainer()
                thread(name = "sandbox-train") {
                    try {
                        val (checkpoint, finalLoss) = t.train(epochs, batchSize) { epoch, step, loss ->
                            val b = Bundle().apply {
                                putInt(TrainingProtocol.KEY_EPOCH, epoch)
                                putInt(TrainingProtocol.KEY_STEP, step)
                                putFloat(TrainingProtocol.KEY_LOSS, loss)
                            }
                            send(TrainingProtocol.MSG_TRAINING_PROGRESS, b)
                        }
                        val b = Bundle().apply {
                            putByteArray(TrainingProtocol.KEY_CHECKPOINT_BYTES, checkpoint)
                            putFloat(TrainingProtocol.KEY_FINAL_LOSS, finalLoss)
                        }
                        send(TrainingProtocol.MSG_TRAINING_COMPLETE, b)
                    } catch (t2: Throwable) {
                        sendError(t2.message ?: t2.javaClass.simpleName)
                    }
                }
            }

            TrainingProtocol.MSG_CANCEL_TRAINING -> {
                trainer?.cancel()
            }

            else -> error("unknown message: ${msg.what}")
        }
    }

    private fun requireTrainer(): SandboxTrainer =
        trainer ?: error("trainer not initialized — send MSG_INIT_MODEL first")

    private fun send(what: Int, data: Bundle) {
        val reply = replyTo ?: return
        val m = Message.obtain(null, what).apply { this.data = data }
        try {
            reply.send(m)
        } catch (_: RemoteException) {
        }
    }

    private fun sendError(message: String) {
        send(TrainingProtocol.MSG_TRAINING_ERROR, Bundle().apply {
            putString(TrainingProtocol.KEY_ERROR_MESSAGE, message)
        })
    }
}

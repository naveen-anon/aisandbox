package com.naveen.aisandbox.training

import android.app.Service
import android.content.Intent
import android.os.*
import com.naveen.aisandbox.ipc.TrainingProtocol
import kotlin.concurrent.thread

/**
 * Lives in the isolated `:training_sandbox` process (see AndroidManifest).
 * This class, and everything it calls, runs under a throwaway OS uid with
 * no network, no access to this app's real files, and no permissions.
 *
 * Communication is a plain Handler/Messenger pair — classic AIDL-less IPC.
 * We deliberately do NOT use a ContentProvider or shared file for this:
 * isolated processes can't resolve this app's ContentProviders anyway, and
 * keeping everything as in-memory Bundle payloads over Binder is what
 * keeps the boundary airtight.
 */
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
                val modelBytes = data.getByteArray(TrainingProtocol.KEY_MODEL_BYTES)
                    ?: error("no model bytes supplied")
                trainer?.close()
                trainer = SandboxTrainer(modelBytes)
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
                // Run off the Binder/main thread of this isolated process so
                // progress messages can still be pumped out during training.
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
            // client process died; nothing to do, isolated process will be torn down too
        }
    }

    private fun sendError(message: String) {
        send(TrainingProtocol.MSG_TRAINING_ERROR, Bundle().apply {
            putString(TrainingProtocol.KEY_ERROR_MESSAGE, message)
        })
    }
}

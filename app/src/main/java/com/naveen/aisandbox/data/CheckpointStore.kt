package com.naveen.aisandbox.data

import android.content.Context
import java.io.File

/**
 * Persists trained checkpoints to this app's private storage
 * (filesDir — never external/shared storage by default). The sandbox
 * process itself never calls this; only the main process does, after
 * receiving checkpoint bytes over Binder.
 */
class CheckpointStore(context: Context) {
    private val dir = File(context.filesDir, "checkpoints").apply { mkdirs() }

    fun save(name: String, bytes: ByteArray): File {
        val f = File(dir, "$name.ckpt")
        f.writeBytes(bytes)
        return f
    }

    fun list(): List<File> = dir.listFiles()?.toList().orEmpty().sortedByDescending { it.lastModified() }

    fun read(name: String): ByteArray? {
        val f = File(dir, "$name.ckpt")
        return if (f.exists()) f.readBytes() else null
    }
}

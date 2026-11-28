package lightrag.utils

import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

internal fun writeStorageFileAtomically(
    file: File,
    text: String,
) {
    val target = file.toPath().toAbsolutePath()
    Files.createDirectories(target.parent)
    val temporary = Files.createTempFile(target.parent, ".lightrag-", ".tmp")
    try {
        Files.writeString(temporary, text)
        Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
    } finally {
        Files.deleteIfExists(temporary)
    }
}

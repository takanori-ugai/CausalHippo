package com.youtu.graphrag.shared.io

import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.Charset
import java.nio.charset.CodingErrorAction
import java.nio.charset.IllegalCharsetNameException
import java.nio.charset.UnsupportedCharsetException

private val candidateEncodings =
    listOf(
        "UTF-8",
        "UTF-8-SIG",
        "GB18030",
        "GBK",
        "Big5",
        "UTF-16",
        "UTF-16LE",
        "UTF-16BE",
        "ISO-8859-1",
    )

fun decodeBytesWithDetection(data: ByteArray): String {
    val detected = detectEncodingFromBom(data)

    val orderedCandidates =
        buildList {
            if (detected != null) {
                add(detected)
            }
            addAll(candidateEncodings)
        }.distinct()

    orderedCandidates.forEach { encoding ->
        decodeStrict(data, encoding)?.let { return it }
    }

    return data.toString(Charsets.UTF_8)
}

private fun decodeStrict(
    data: ByteArray,
    encoding: String,
): String? {
    val resolvedEncoding =
        if (encoding.equals("UTF-8-SIG", ignoreCase = true)) {
            "UTF-8"
        } else {
            encoding
        }

    return try {
        val decoder =
            Charset
                .forName(resolvedEncoding)
                .newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)

        val decoded = decoder.decode(ByteBuffer.wrap(data)).toString()
        decoded.removePrefix("\uFEFF")
    } catch (_: CharacterCodingException) {
        null
    } catch (_: UnsupportedCharsetException) {
        null
    } catch (_: IllegalCharsetNameException) {
        null
    }
}

private fun detectEncodingFromBom(data: ByteArray): String? {
    if (data.size >= 3 && data[0] == 0xEF.toByte() && data[1] == 0xBB.toByte() && data[2] == 0xBF.toByte()) {
        return "UTF-8"
    }

    if (data.size >= 2 && data[0] == 0xFF.toByte() && data[1] == 0xFE.toByte()) {
        return "UTF-16LE"
    }

    if (data.size >= 2 && data[0] == 0xFE.toByte() && data[1] == 0xFF.toByte()) {
        return "UTF-16BE"
    }

    return null
}

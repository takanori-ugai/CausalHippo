package com.youtu.graphrag.shared.io

import com.youtu.graphrag.shared.io.decodeBytesWithDetection
import kotlin.test.Test
import kotlin.test.assertEquals

class EncodingDecodersTest {
    @Test
    fun `decodes utf8 text`() {
        val data = "plain utf-8 text".encodeToByteArray()

        assertEquals("plain utf-8 text", decodeBytesWithDetection(data))
    }

    @Test
    fun `decodes utf16le text with bom`() {
        val text = "hello world"
        val bom = byteArrayOf(0xFF.toByte(), 0xFE.toByte())
        val payload = text.toByteArray(Charsets.UTF_16LE)
        val data = bom + payload

        assertEquals(text, decodeBytesWithDetection(data))
    }

    @Test
    fun `handles garbage input gracefully`() {
        val data = byteArrayOf(0x80.toByte(), 0x81.toByte())
        // Should not throw exception
        decodeBytesWithDetection(data)
    }
}

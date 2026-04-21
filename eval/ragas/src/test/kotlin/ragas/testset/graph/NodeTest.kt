package ragas.testset.graph

import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class NodeTest {
    @Test
    fun nodeConstructorIncludesDuplicateKeysInErrorMessage() {
        val error =
            assertFailsWith<IllegalArgumentException> {
                Node(
                    type = NodeType.DOCUMENT,
                    properties =
                        mutableMapOf(
                            "Page_Content" to "alpha",
                            "page_content" to "beta",
                            "Another_Key" to "1",
                            "another_key" to "2",
                        ),
                )
            }
        val message = error.message.orEmpty()
        assertTrue(message.contains("Duplicate property keys when normalized to lowercase"))
        assertTrue(message.contains("page_content"))
        assertTrue(message.contains("another_key"))
    }
}

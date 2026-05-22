package shared.rag.unified

import causalhippo.CausalHippoQueryParam
import causalhippo.CausalHippoRAG
import causalrag.CausalRAG
import causalrag.CausalRagRunResult
import com.microsoft.graphrag.GraphRAG
import hipporag.HippoRAG
import hipporag.utils.QuerySolution
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.runBlocking
import lightrag.core.LightRAG
import pathrag.PathRAG
import kotlin.test.Test
import kotlin.test.assertEquals
import com.microsoft.graphrag.QueryParam as GraphQueryParam
import com.microsoft.graphrag.query.QueryResult as GraphQueryResult
import hipporag.QueryParam as HippoQueryParam
import lightrag.core.QueryParam as LightQueryParam
import lightrag.core.QueryResult as LightQueryResult
import pathrag.base.QueryParam as PathQueryParam

class UnifiedModeMappingTest {
    @Test
    fun `graph adapter maps LOCAL to local`() =
        runBlocking {
            val delegate = mockk<GraphRAG>()
            coEvery { delegate.aquery(any(), any()) } returns GraphQueryResult(answer = "ok", context = emptyList())

            val adapter = GraphRagUnifiedAdapter(delegate)
            adapter.aquery("q", UnifiedQuery(mode = UnifiedMode.LOCAL))

            coVerify(exactly = 1) {
                delegate.aquery(
                    "q",
                    withArg<GraphQueryParam> { param ->
                        assertEquals("local", param.mode)
                    },
                )
            }
        }

    @Test
    fun `light adapter maps NAIVE to naive`() =
        runBlocking {
            val delegate = mockk<LightRAG>()
            coEvery { delegate.aquery(any(), any()) } returns LightQueryResult(content = "ok")

            val adapter = LightRagUnifiedAdapter(delegate)
            adapter.aquery("q", UnifiedQuery(mode = UnifiedMode.NAIVE))

            coVerify(exactly = 1) {
                delegate.aquery(
                    "q",
                    withArg<LightQueryParam> { param ->
                        assertEquals("naive", param.mode)
                    },
                )
            }
        }

    @Test
    fun `path adapter maps GLOBAL to global`() =
        runBlocking {
            val delegate = mockk<PathRAG>()
            coEvery { delegate.aquery(any(), any()) } returns "ctx"

            val adapter = PathRagUnifiedAdapter(delegate)
            adapter.aquery(
                "q",
                UnifiedQuery(
                    mode = UnifiedMode.GLOBAL,
                    includeAnswer = false,
                    includeContext = true,
                ),
            )

            coVerify(exactly = 1) {
                delegate.aquery(
                    "q",
                    withArg<PathQueryParam> { param ->
                        assertEquals("global", param.mode)
                    },
                )
            }
        }

    @Test
    fun `hippo adapter maps DPR to dpr`() =
        runBlocking {
            val delegate = mockk<HippoRAG>()
            coEvery { delegate.aquery(any(), any()) } returns QuerySolution(question = "q", docs = emptyList(), answer = "ok")

            val adapter = HippoRagUnifiedAdapter(delegate)
            adapter.aquery("q", UnifiedQuery(mode = UnifiedMode.DPR))

            coVerify(exactly = 1) {
                delegate.aquery(
                    "q",
                    withArg<HippoQueryParam> { param ->
                        assertEquals("dpr", param.mode)
                    },
                )
            }
        }

    @Test
    fun `causal adapter maps maxPaths extra`() =
        runBlocking {
            val delegate = mockk<CausalRAG>()
            coEvery { delegate.aquery(any(), any()) } returns
                CausalRagRunResult(answer = "ok", context = emptyList(), causalPaths = emptyList())

            val adapter = CausalRagUnifiedAdapter(delegate)
            adapter.aquery(
                "q",
                UnifiedQuery(
                    mode = UnifiedMode.CAUSAL,
                    extras = mapOf("maxPaths" to 7),
                ),
            )

            coVerify(exactly = 1) {
                delegate.aquery(
                    "q",
                    withArg<causalrag.QueryParam> { param ->
                        assertEquals(7, param.maxPaths)
                    },
                )
            }
        }

    @Test
    fun `causal hippo adapter maps maxPaths extra`() =
        runBlocking {
            val delegate = mockk<CausalHippoRAG>()
            coEvery { delegate.aquery(any(), any()) } returns
                CausalRagRunResult(answer = "ok", context = emptyList(), causalPaths = emptyList())

            val adapter = CausalHippoRagUnifiedAdapter(delegate)
            adapter.aquery(
                "q",
                UnifiedQuery(
                    mode = UnifiedMode.CAUSAL,
                    extras = mapOf("maxPaths" to 9),
                ),
            )

            coVerify(exactly = 1) {
                delegate.aquery(
                    "q",
                    withArg<CausalHippoQueryParam> { param ->
                        assertEquals(9, param.maxPaths)
                    },
                )
            }
        }
}

package lightrag.examples

import io.github.oshai.kotlinlogging.KotlinLogging
import javafx.application.Application
import javafx.geometry.Insets
import javafx.scene.Group
import javafx.scene.Scene
import javafx.scene.control.Button
import javafx.scene.control.Label
import javafx.scene.control.Tooltip
import javafx.scene.input.ScrollEvent
import javafx.scene.layout.BorderPane
import javafx.scene.layout.HBox
import javafx.scene.paint.Color
import javafx.scene.shape.Circle
import javafx.scene.shape.Line
import javafx.scene.text.Text
import javafx.stage.Stage
import kotlinx.coroutines.runBlocking
import lightrag.di.appModule
import lightrag.services.StorageManager
import org.koin.core.context.startKoin
import java.util.UUID

/**
 * Minimal JavaFX graph visualizer wired to the current GraphStorage.
 * Run with: ./gradlew run -PmainClass=lightrag.examples.GraphStorageVisualizerFx
 */
class GraphStorageVisualizerFx : Application() {
    private lateinit var storageManager: StorageManager
    private val pane = Group()
    private val logger = KotlinLogging.logger {}
    private val minScale = 0.2
    private val maxScale = 5.0

    override fun start(stage: Stage) {
        val koin = startKoin { modules(appModule) }.koin
        storageManager = koin.get()

        val reload = Button("Reload Graph")
        val status = Label("Loading graph…")
        val controls =
            HBox(10.0, reload, Text("Scroll to zoom | Nodes sized by degree | Click node for label")).apply {
                padding = Insets(8.0)
            }

        reload.setOnAction { loadAndRender(status) }

        val root = BorderPane()
        root.top = controls
        root.center = pane
        root.bottom = status.apply { padding = Insets(6.0) }

        val scene = Scene(root, 1200.0, 800.0, Color.web("#0c0c0f"))
        val zoomHandler = fun(event: ScrollEvent) {
            if (event.deltaY == 0.0) return
            // Treat trackpad/wheel up as zoom in.
            val zoomIn = event.deltaY > 0
            val scaleFactor = if (zoomIn) 1.1 else 0.9
            val oldScale = pane.scaleX
            val newScale = (oldScale * scaleFactor).coerceIn(minScale, maxScale)

            val pivotX = scene.width / 2.0
            val pivotY = scene.height / 2.0

            val f = newScale / oldScale

            pane.scaleX = newScale
            pane.scaleY = newScale
            pane.translateX = pane.translateX * f + (1 - f) * pivotX
            pane.translateY = pane.translateY * f + (1 - f) * pivotY
            event.consume()
        }
        scene.setOnScroll(zoomHandler)
        pane.setOnScroll(zoomHandler)

        stage.title = "LightRAG Graph Storage Visualizer"
        stage.scene = scene
        stage.show()

        loadAndRender(status)
    }

    private fun loadAndRender(status: Label) {
        status.text = "Loading graph…"
        pane.children.clear()

        val graphData =
            runBlocking {
                storageManager.initialize()
                val nodes = storageManager.chunkEntityRelationGraph.getAllNodes()
                val edges = storageManager.chunkEntityRelationGraph.getAllEdges()
                GraphData(nodes, edges)
            }

        if (graphData.nodes.isEmpty()) {
            status.text = "No nodes available in graph storage."
            return
        }

        val visualNodes = buildVisualNodes(graphData.nodes)
        val aliasMap = buildAliasLookup(visualNodes, graphData.nodes)

        graphData.edges.forEach { edge ->
            val src = edge["source"] ?: edge["src_id"] ?: edge["source_id"] ?: edge["src"]
            val tgt = edge["target"] ?: edge["tgt_id"] ?: edge["target_id"] ?: edge["tgt"]
            if (src == null || tgt == null) return@forEach
            val srcNode = aliasMap[src.toString()] ?: return@forEach
            val tgtNode = aliasMap[tgt.toString()] ?: return@forEach
            pane.children.add(
                Line(srcNode.x, srcNode.y, tgtNode.x, tgtNode.y).apply {
                    stroke = Color.web("#90caf9", 0.7)
                    strokeWidth = 1.5
                },
            )
        }

        visualNodes.forEach { node ->
            pane.children.add(
                Circle(node.x, node.y, node.radius, Color.web("#4fc3f7", 0.85)).apply {
                    stroke = Color.web("#e1f5fe")
                    strokeWidth = 1.5
                    Tooltip.install(this, Tooltip("${node.label} (deg ${node.degree})"))
                    setOnMouseClicked { status.text = "Selected: ${node.label} (id ${node.id})" }
                },
            )
            pane.children.add(
                Text(node.x + node.radius + 4, node.y, node.label.take(30)).apply {
                    fill = Color.LIGHTGRAY
                    setOnMouseClicked { status.text = "Selected: ${node.label} (id ${node.id})" }
                },
            )
        }

        status.text = "Loaded ${visualNodes.size} nodes / ${graphData.edges.size} edges"
        logger.info { status.text }
    }

    private fun buildVisualNodes(nodes: List<Map<String, Any>>): List<VisualNode> {
        val width = 1000.0
        val height = 700.0
        val radius = minOf(width, height) / 2.5

        return nodes.mapIndexed { idx, node ->
            val id =
                (
                    node["id"]
                        ?: node["entity_id"]
                        ?: node["entity_name"]
                        ?: node["name"]
                        ?: UUID.randomUUID()
                ).toString()
            val label = (node["entity_name"] ?: node["label"] ?: node["name"] ?: id).toString()
            val degree = (node["degree"] as? Number)?.toInt() ?: 1
            val angle = (idx.toDouble() / nodes.size) * Math.PI * 2
            val x = width / 2 + radius * kotlin.math.cos(angle)
            val y = height / 2 + radius * kotlin.math.sin(angle)
            VisualNode(id, label, degree, x, y, radius = 8.0 + degree.coerceAtMost(8))
        }
    }

    private fun buildAliasLookup(
        visualNodes: List<VisualNode>,
        rawNodes: List<Map<String, Any>>,
    ): Map<String, VisualNode> {
        val aliasMap = mutableMapOf<String, VisualNode>()
        visualNodes.forEachIndexed { idx, vis ->
            val raw = rawNodes.getOrNull(idx) ?: emptyMap()
            val aliases =
                listOfNotNull(
                    vis.id,
                    raw["id"],
                    raw["entity_id"],
                    raw["entity_name"],
                    raw["name"],
                    raw["label"],
                ).map { it.toString() }
            aliases.forEach { aliasMap[it] = vis }
        }
        return aliasMap
    }

    data class GraphData(
        val nodes: List<Map<String, Any>>,
        val edges: List<Map<String, Any>>,
    )

    data class VisualNode(
        val id: String,
        val label: String,
        val degree: Int,
        val x: Double,
        val y: Double,
        val radius: Double,
    )
}

fun main(args: Array<String>) {
    Application.launch(GraphStorageVisualizerFx::class.java, *args)
}

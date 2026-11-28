package pathrag.operate

/** Scores the one- to three-hop routes connecting one ordered seed pair. */
internal fun scorePathFlows(
    source: String,
    target: String,
    paths: List<List<String>>,
): List<Pair<List<String>, Double>> {
    val successors = mutableMapOf<String, MutableSet<String>>()
    for (path in paths) {
        path.zipWithNext().forEach { (from, to) -> successors.getOrPut(from) { mutableSetOf() }.add(to) }
    }
    val firstHops = successors[source].orEmpty()
    if (firstHops.isEmpty()) return emptyList()
    val edgeTotals = mutableMapOf<Pair<String, String>, Double>()

    fun record(
        from: String,
        neighbor: String,
        flow: Double,
    ) {
        val edge = from to neighbor
        edgeTotals[edge] = edgeTotals.getOrDefault(edge, 0.0) + flow
    }

    val decay = 0.8
    val threshold = 0.3
    val firstFlow = 1.0 / firstHops.size
    for (first in firstHops.sorted()) {
        record(source, first, firstFlow)
        if (first == target || firstFlow <= threshold) continue
        val secondHops = successors.getValue(first)
        val secondFlow = decay * firstFlow / secondHops.size
        for (second in secondHops) {
            record(first, second, secondFlow)
            // A third-hop arrival on this edge counts toward reliability but cannot fund a fourth hop.
            if (second == target || secondFlow <= threshold) continue
            val thirdHops = successors.getValue(second)
            val thirdFlow = decay * secondFlow / thirdHops.size
            for (third in thirdHops) record(second, third, thirdFlow)
        }
    }
    return paths.map { path ->
        path to path.zipWithNext().sumOf { edge -> edgeTotals.getOrDefault(edge, 0.0) } / (path.size - 1)
    }
}

/** Breaks equal-score ties within a seed pair independently of storage iteration order. */
internal fun compareNodeSequences(
    left: List<String>,
    right: List<String>,
): Int {
    for (index in 0 until minOf(left.size, right.size)) {
        val order = left[index].compareTo(right[index])
        if (order != 0) return order
    }
    return left.size.compareTo(right.size)
}

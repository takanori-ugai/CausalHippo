rootProject.name = "causalrag"
includeBuild("eval/ragas")
includeBuild("eval/bertscore") {
    dependencySubstitution {
        substitute(module("io.github.ugaikit:bertscore")).using(project(":"))
    }
}

import org.jlleitschuh.gradle.ktlint.reporter.ReporterType

plugins {
    kotlin("jvm") version "2.4.0"
    application
    id("com.gradleup.shadow") version "9.4.1"
    kotlin("plugin.serialization") version "2.3.20"
    id("org.jlleitschuh.gradle.ktlint") version "14.2.0"
    id("io.gitlab.arturbosch.detekt") version "1.23.8"
}

detekt {
    buildUponDefaultConfig = true
    config.setFrom(files("config/detekt.yml"))
}

group = "com.causalrag"
version = "0.0.1"

val ktorVersion = "3.4.1"
val osName = System.getProperty("os.name").lowercase()
val archName = System.getProperty("os.arch").lowercase()
val networkAnalysisVersion = "1.3.0"
val jgraphtVersion = "1.5.2"
val javafxVersion = "21.0.5"
val javafxPlatform =
    when {
        osName.contains("mac") && archName.contains("aarch64") -> "mac-aarch64"
        osName.contains("mac") -> "mac"
        osName.contains("win") && archName.contains("aarch64") -> "win-aarch64"
        osName.contains("win") -> "win"
        archName.contains("aarch64") || archName.contains("arm64") -> "linux-aarch64"
        else -> "linux"
    }

application {
    val requestedMain =
        if (project.hasProperty("mainClass")) {
            project.property("mainClass") as String
        } else {
            null
        }
    mainClass.set(requestedMain ?: "causalrag.examples.CliKt")
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(platform("io.ktor:ktor-bom:$ktorVersion"))
    implementation("io.ktor:ktor-server-core")
    implementation("io.ktor:ktor-server-netty")
    implementation("io.ktor:ktor-server-content-negotiation")
    implementation("io.ktor:ktor-serialization-kotlinx-json")
    implementation("io.ktor:ktor-server-swagger")
    implementation("io.ktor:ktor-server-cors")
    implementation("ch.qos.logback:logback-classic:1.5.32")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
    implementation("io.github.oshai:kotlin-logging-jvm:8.0.01")
    implementation("com.github.haifengl:smile-core:4.4.2")
    implementation("gg.jte:jte-kotlin:3.2.4")
    implementation("org.apache.commons:commons-csv:1.14.1")
    implementation("org.apache.pdfbox:pdfbox:3.0.7")
    implementation("io.ragas:ragas-kotlin:0.0.1")
    implementation("io.github.ugaikit:bertscore:0.1.0")

    // LangChain4j dependencies
    implementation("dev.langchain4j:langchain4j:1.15.0")
    implementation("dev.langchain4j:langchain4j-open-ai:1.15.0")
    implementation("dev.langchain4j:langchain4j-open-ai-official:1.15.0-beta25")
    implementation("dev.langchain4j:langchain4j-ollama:1.15.0")
    implementation("dev.langchain4j:langchain4j-community-neo4j:1.15.0-beta25")

    implementation("com.auth0:java-jwt:4.5.1")
    // JTokkit
    implementation("com.knuddels:jtokkit:1.1.0")

    // MongoDB
    implementation("org.mongodb:mongodb-driver-kotlin-coroutine:5.8.0")
    implementation("org.mongodb:bson-kotlinx:5.8.0")
    implementation("org.neo4j.driver:neo4j-java-driver:6.0.3")
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit:2.3.21")
    testImplementation("io.mockk:mockk:1.14.9")

    implementation("org.openjfx:javafx-base:$javafxVersion:$javafxPlatform")
    implementation("org.openjfx:javafx-graphics:$javafxVersion:$javafxPlatform")
    implementation("org.openjfx:javafx-controls:$javafxVersion:$javafxPlatform")

    implementation("com.fasterxml.jackson.module:jackson-module-kotlin:2.21.3")
    implementation("org.jgrapht:jgrapht-core:$jgraphtVersion")
    // Keep Hadoop before parquet-floor on classpath: parquet-floor bundles a stub FSDataInputStream.
    implementation("org.apache.hadoop:hadoop-client-api:3.5.0")
    runtimeOnly("org.apache.hadoop:hadoop-client-runtime:3.5.0")
    implementation("blue.strategic.parquet:parquet-floor:2.1")
    implementation("info.picocli:picocli:4.7.6")
    implementation("nl.cwts:networkanalysis:$networkAnalysisVersion")
}

tasks {
    withType<Test> {
        jvmArgs("-XX:+EnableDynamicAgentLoading")
        testLogging {
            events("passed", "skipped", "failed", "standardOut", "standardError")
            showStandardStreams = true
        }
    }

    // Separate task for scriptable/CLI runs; keeps `run` intact for IDE defaults.
    val execute by registering(JavaExec::class) {
        group = "application"
        mainClass.set(application.mainClass)
        classpath = sourceSets.main.get().runtimeClasspath
    }

    shadowJar {
        isZip64 = true
    }
}

ktlint {
    version.set("1.8.0")
    verbose.set(true)
    outputToConsole.set(true)
    coloredOutput.set(true)
    reporters {
        reporter(ReporterType.CHECKSTYLE)
        reporter(ReporterType.JSON)
        reporter(ReporterType.HTML)
    }
    filter {
        exclude("**/style-violations.kt")
    }
}

# Installation (Kotlin)

## Build from source

```bash
./gradlew build
```

## Run tests

```bash
./gradlew test
```

## Optional: run CLI scaffold

```bash
./gradlew run --args="help"
./gradlew run --args="status"
./gradlew run --args="backends"
```

## Use as a dependency

If you publish artifacts in your environment, add:

```kotlin
dependencies {
    implementation("io.github.ragas:ragas-kotlin:<version>")
}
```

If you are developing in this repository directly, no package install step is required.

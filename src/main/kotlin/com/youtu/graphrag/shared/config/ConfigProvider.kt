package com.youtu.graphrag.shared.config

object ConfigProvider {
    @Volatile
    private var configInstance: ConfigManager? = null

    @Synchronized
    fun getConfig(configPath: String? = null): ConfigManager {
        val current = configInstance
        if (current != null) {
            return current
        }

        return ConfigManager(configPath).also { configInstance = it }
    }

    @Synchronized
    fun reloadConfig(configPath: String? = null): ConfigManager = ConfigManager(configPath).also { configInstance = it }
}

package com.example.streaming

import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

data class SensorReading(val sensorId: String, val value: Double, val timestamp: Long)

class DataStream(private val sensorId: String) {

    fun readings(intervalMs: Long = 1000): Flow<SensorReading> = flow {
        while (true) {
            val reading = SensorReading(
                sensorId = sensorId,
                value = Math.random() * 100,
                timestamp = System.currentTimeMillis()
            )
            emit(reading)
            delay(intervalMs)
        }
    }

    fun bounded(count: Int): Flow<SensorReading> = flow {
        repeat(count) { i ->
            emit(SensorReading(sensorId, Math.random() * 100, System.currentTimeMillis()))
            delay(100)
        }
    }
}

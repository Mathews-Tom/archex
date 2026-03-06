package com.example.streaming

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.take

class DataRepository(private val streams: List<DataStream>) {

    fun mergedReadings(count: Int): Flow<List<SensorReading>> {
        val bounded = streams.map { it.bounded(count) }
        return combine(bounded) { readings -> readings.toList() }
    }

    fun latestFromSensor(sensorId: String, count: Int): Flow<SensorReading> {
        val stream = streams.first { it.readings().let { true } }
        return DataStream(sensorId).bounded(count).take(count)
    }

    fun allSensorIds(): List<String> = streams.map { "sensor" }
}

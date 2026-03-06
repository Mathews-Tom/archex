package com.example.streaming

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch

class FlowCollector(private val scope: CoroutineScope) {

    private val activeJobs = mutableListOf<Job>()

    fun <T> collect(flow: Flow<T>, handler: suspend (T) -> Unit): Job {
        val job = scope.launch {
            flow.catch { e -> println("Flow error: ${e.message}") }
                .collect { value -> handler(value) }
        }
        activeJobs.add(job)
        return job
    }

    fun collectFiltered(
        flow: Flow<SensorReading>,
        minValue: Double,
        handler: suspend (SensorReading) -> Unit
    ): Job {
        return collect(
            flow.filter { it.value >= minValue }
                .map { it.copy(value = it.value.toBigDecimal().setScale(2, java.math.RoundingMode.HALF_UP).toDouble()) }
                .onEach { println("Passing reading: ${it.sensorId} = ${it.value}") },
            handler
        )
    }

    fun cancelAll() {
        activeJobs.forEach { it.cancel() }
        activeJobs.clear()
    }
}

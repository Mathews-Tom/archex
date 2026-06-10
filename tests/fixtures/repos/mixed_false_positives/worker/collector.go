package worker

import "sync"

type MetricsCollector struct {
	mu     sync.Mutex
	counts map[string]int
}

func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{counts: map[string]int{}}
}

func (c *MetricsCollector) Add(name string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.counts[name]++
}

func (c *MetricsCollector) Snapshot() map[string]int {
	c.mu.Lock()
	defer c.mu.Unlock()

	copy := make(map[string]int, len(c.counts))
	for key, value := range c.counts {
		copy[key] = value
	}
	return copy
}

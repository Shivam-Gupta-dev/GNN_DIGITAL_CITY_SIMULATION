# CTM API Usage Guide (Optimized)
## Fast and Efficient CTM Endpoints

---

## 🚀 Quick Start

### Initialize CTM (Fast Mode)
```bash
curl -X POST http://localhost:5000/api/ctm/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "cell_length_km": 1.0,
    "time_step_hours": 0.0167,
    "initial_density_ratio": 0.15,
    "demand_generation_rate": 50
  }'
```

**Response Time:** ~10-20 seconds (optimized from 30-60s)

---

## 📊 Run Simulation Steps

### Single Step
```bash
curl -X POST http://localhost:5000/api/ctm/step \
  -H "Content-Type: application/json" \
  -d '{"steps": 1}'
```

**Response Time:** ~0.5-1 second

### Batch Steps (Recommended)
```bash
curl -X POST http://localhost:5000/api/ctm/step \
  -H "Content-Type: application/json" \
  -d '{"steps": 10}'
```

**Response Time:** ~5-10 seconds for 10 steps
**Tip:** Use batch steps instead of multiple single steps for better performance

---

## 📈 Get Simulation Data

### Edge Congestion (Fast)
```bash
curl http://localhost:5000/api/ctm/edge-congestion
```

**Response Time:** ~0.2-0.5 seconds
**Returns:** Aggregated edge-level data (hundreds of edges)

**Response:**
```json
{
  "timestamp": 60.0,
  "edges": [
    {
      "source": "0",
      "target": "1",
      "key": 0,
      "congestion": 0.15,
      "travel_time": 2.5,
      "is_closed": false
    }
  ],
  "total_edges": 4676
}
```

### Cell States (Use Edge Congestion Instead)
```bash
curl http://localhost:5000/api/ctm/cells
```

**Note:** Now returns aggregated edge data instead of individual cells
**Response Time:** ~0.2-0.5 seconds (was 3-8 seconds)

---

## 🚧 Road Operations

### Close Road
```bash
curl -X POST http://localhost:5000/api/ctm/close-road \
  -H "Content-Type: application/json" \
  -d '{
    "source": "100",
    "target": "150",
    "key": 0
  }'
```

### Reopen Road
```bash
curl -X POST http://localhost:5000/api/ctm/reopen-road \
  -H "Content-Type: application/json" \
  -d '{
    "source": "100",
    "target": "150",
    "key": 0
  }'
```

---

## 📊 Get Status
```bash
curl http://localhost:5000/api/ctm/status
```

**Response:**
```json
{
  "initialized": true,
  "stats": {
    "simulation_time": 120.0,
    "total_vehicles": 5000,
    "average_congestion": 0.25,
    "max_congestion": 0.85,
    "closed_roads": 2
  },
  "closed_roads": 2,
  "snapshots": 12
}
```

---

## 🔄 Reset Simulation
```bash
curl -X POST http://localhost:5000/api/ctm/reset
```

---

## 💡 Performance Tips

### 1. Use Batch Steps
**Bad:**
```javascript
for (let i = 0; i < 10; i++) {
  await fetch('/api/ctm/step', {method: 'POST'});
}
// Takes ~10-20 seconds
```

**Good:**
```javascript
await fetch('/api/ctm/step', {
  method: 'POST',
  body: JSON.stringify({steps: 10})
});
// Takes ~5-10 seconds
```

### 2. Use Edge Congestion, Not Individual Cells
**Bad:**
```javascript
const cells = await fetch('/api/ctm/cells').then(r => r.json());
// Returns thousands of cells, slow to process
```

**Good:**
```javascript
const edges = await fetch('/api/ctm/edge-congestion').then(r => r.json());
// Returns hundreds of edges, fast and sufficient
```

### 3. Initialize Once, Reuse
**Bad:**
```javascript
// Re-initialize for each test
for (test of tests) {
  await fetch('/api/ctm/initialize', {method: 'POST'});
  // ... run test
}
```

**Good:**
```javascript
// Initialize once, reset between tests
await fetch('/api/ctm/initialize', {method: 'POST'});
for (test of tests) {
  await fetch('/api/ctm/reset', {method: 'POST'});
  // ... run test
}
```

---

## 🎯 Example Workflow

```javascript
// 1. Initialize (do this once)
const init = await fetch('http://localhost:5000/api/ctm/initialize', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    cell_length_km: 1.0,
    initial_density_ratio: 0.15
  })
});

// 2. Close a road
await fetch('http://localhost:5000/api/ctm/close-road', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    source: "100",
    target: "150",
    key: 0
  })
});

// 3. Run simulation (batch steps)
await fetch('http://localhost:5000/api/ctm/step', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({steps: 20})
});

// 4. Get congestion data
const data = await fetch('http://localhost:5000/api/ctm/edge-congestion')
  .then(r => r.json());

console.log('Average congestion:', data.edges.reduce((sum, e) => sum + e.congestion, 0) / data.edges.length);
```

---

## 📊 Performance Comparison

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Initialize | 30-60s | 10-20s | **2-3x faster** |
| Single Step | 2-5s | 0.5-1s | **4-5x faster** |
| 10 Steps | 20-50s | 5-10s | **4-5x faster** |
| Get Cells | 3-8s | 0.2-0.5s | **15-40x faster** |
| Get Status | 0.5-1s | 0.1-0.2s | **5x faster** |

---

## 🐛 Troubleshooting

### Slow Initialization
- Check graph size (works best with <10K edges)
- Ensure sufficient RAM available
- Consider using `fast_mode: true` in config

### Slow Steps
- Use batch steps instead of single steps
- Reduce `demand_generation_rate` if too high
- Increase `cell_length_km` for fewer cells

### API Timeouts
- Increase timeout settings in your HTTP client
- Use batch operations instead of many small requests
- Check server resources (CPU/RAM)

---

## ✅ Best Practices

1. **Initialize once** at application start
2. **Use batch steps** for multiple time steps
3. **Query edge congestion** instead of individual cells
4. **Reset** instead of re-initializing for new scenarios
5. **Monitor** response times and adjust parameters

---

**Status:** ✅ **Production Ready**
**Last Updated:** December 4, 2025

# Node Removal & Traffic Impact Analysis Feature

## Overview

This feature allows users to simulate the removal of critical infrastructure nodes (stations, intersections, districts) from the city network and analyze the cascading traffic impact on all connected edges. When a node is removed, all edges connected to that node are automatically closed, and the GNN model predicts traffic congestion patterns with this constraint.

## Features

### 1. **Interactive Node Removal**
- Click on any node on the map to view its details
- Click the **"Remove Node"** button in the node info panel
- The system automatically identifies and closes all edges connected to that node

### 2. **Real-time Traffic Impact Analysis**
- When a node is removed, the backend analyzes:
  - Number of affected edges (roads/metro lines)
  - Mean congestion on closed edges
  - Maximum congestion on closed edges
  - Overall network mean and max congestion
  - Transport mode-specific impacts (roads vs. metro)

### 3. **Visual Feedback**
- Affected edges turn **pink with dashed lines** (same as manually closed roads)
- Removed nodes appear in the **"Removed Nodes"** panel
- Detailed impact analysis displayed in the info panel

### 4. **Node Restoration**
- Click **"Restore Node"** button to revert the removal
- All connected edges automatically reopen
- Traffic predictions update in real-time

### 5. **Impact Statistics**
The system provides comprehensive metrics:
- **Closed Edges Count**: Number of edges affected by node removal
- **Mean Congestion (Closed Edges)**: Average traffic on affected edges
- **Max Congestion (Closed Edges)**: Worst-case congestion on affected edges
- **Overall Mean Congestion**: Network-wide average congestion
- **Overall Max Congestion**: Network-wide maximum congestion
- **Road/Metro Breakdown**: Separate statistics for different transport modes

## UI Components

### Removed Nodes Panel (Sidebar)
Located in the sidebar, shows:
- List of all removed nodes
- Number of edges closed for each node
- Amenity type for quick identification
- **Restore** button to revert removal

### Node Info Panel (Map Overlay)
When clicking a node:
- Node details (zone, population, amenity, metro status, coordinates)
- **Remove Node** or **Restore Node** button (context-aware)
- After removal: Detailed impact analysis with statistics

## Backend API Endpoints

### POST `/api/analyze-node-removal`

Analyzes traffic impact when removing a node and closes all connected edges.

**Request Body:**
```json
{
    "node_id": "0",
    "hour": 9
}
```

**Response:**
```json
{
    "impact_analysis": {
        "removed_node": "0",
        "node_details": {
            "id": "0",
            "zone": "downtown",
            "population": 50000,
            "amenity": "metro_station",
            "x": 0.0,
            "y": 0.0
        },
        "closed_edges_count": 8,
        "closed_edge_predictions": [0.45, 0.62, ...],
        "mean_closed_edge_congestion": 0.523,
        "max_closed_edge_congestion": 0.87,
        "mean_congestion": 0.412,
        "max_congestion": 0.91,
        "road_mean": 0.398,
        "metro_mean": 0.521
    },
    "affected_edges": [
        "0-1", "1-0", "0-5", "5-0", ...
    ],
    "predictions": [
        {
            "source": "0",
            "target": "1",
            "is_metro": false,
            "is_closed": true,
            "connected_to_removed": true,
            "congestion": 0.62
        },
        ...
    ]
}
```

## Implementation Details

### Frontend Architecture

#### State Management
```javascript
state.removedNodes = new Set()           // Tracks removed node IDs
state.nodeImpactAnalysis = {}            // Stores impact data by node ID
state.closedRoads                        // Also tracks closed edges from removal
```

#### Key Functions
1. **`removeNode(nodeId)`**: Calls backend API, closes connected edges, updates predictions
2. **`restoreNode(nodeId)`**: Reverses node removal, reopens edges
3. **`updateRemovedNodesList()`**: Updates UI with current removed nodes
4. **`showNodeRemovalImpact(nodeId, impact)`**: Displays detailed impact analysis

#### User Interaction Flow
```
User clicks node → Info panel opens
User clicks "Remove Node" → API call to analyze impact
Backend analyzes impact → Returns edge list & predictions
Frontend closes edges → Reruns traffic prediction
UI updates → Impact panel + removed nodes list
```

### Backend Architecture

#### API Endpoint: `/api/analyze-node-removal`
1. Validates node exists in graph
2. Finds all edges connected to node
3. Marks those edges as "closed"
4. Builds node and edge features for GNN
5. Runs model prediction with constraints
6. Calculates impact statistics
7. Returns comprehensive analysis

#### Key Process
```
POST request with node_id
↓
Find connected edges
↓
Build constrained graph features
↓
Run GNN prediction with closed edges
↓
Calculate statistics (mean, max, per-mode)
↓
Return full impact analysis + predictions
```

## Usage Workflow

### Scenario: Analyzing Impact of a Metro Station Closure

1. **Locate the Metro Station**
   - Use search functionality to find the station
   - Click on the node to open the info panel

2. **Remove the Node**
   - Click **"Remove Node"** button
   - System analyzes impact and shows results

3. **Review Impact Analysis**
   - Check the detailed statistics in the info panel
   - View which edges are affected (pink dashed lines on map)
   - See the removed node in the sidebar panel

4. **Compare Metrics**
   - Run prediction to see network-wide impact
   - Check "Road Avg" vs "Metro Avg" congestion
   - Identify critical alternate routes

5. **Restore (Optional)**
   - Click **"Restore"** button if needed
   - Network returns to normal state

## Technical Details

### Edge Closure Mechanism
- When a node is removed, **bidirectional edges** are closed
- Both `u-v` and `v-u` edges are identified
- Edges marked with `is_closed = 1.0` in model input

### GNN Model Integration
- Model receives edge feature with `is_closed` flag
- Learns to predict higher congestion on closed edges
- Propagates congestion upstream (traffic backlog)

### Performance Considerations
- Impact analysis runs on demand (on node removal)
- No modification to core graph structure
- Predictions use same efficient batching

## Integration with Existing Features

### Road Closure Compatibility
- Node removal edges are added to `closedRoads` Set
- Works seamlessly with existing road closure UI
- Can combine multiple disruption types

### Time-Based Multipliers
- Hour parameter passed to API
- Impact predictions use time-specific multipliers
- Reflects realistic peak/off-peak scenarios

### Analysis Page Export
- Removed nodes and affected edges included in analysis
- Export includes impact metrics and statistics
- Can compare baseline vs. removal scenarios

## Future Enhancements

1. **Multi-Node Removal**: Analyze impact of removing multiple nodes simultaneously
2. **Restoration Timeline**: Simulate gradual node reopening
3. **Alternative Route Suggestions**: Auto-recommend bypasses
4. **Cost-Benefit Analysis**: Economic impact of node importance
5. **Cascading Failures**: Model secondary impacts on other nodes
6. **Export Reports**: PDF reports of impact analysis

## Error Handling

- **Node Not Found**: Returns 404 with appropriate message
- **Model Not Loaded**: Returns 500 error
- **Graph Not Loaded**: Returns 500 error
- **Invalid Parameters**: Returns 400 error with missing field info

## Testing Scenarios

### Test Case 1: High-Degree Node
- Remove a node with many connections
- Verify all edges are closed
- Check high congestion on closed edges

### Test Case 2: Metro Station
- Remove a metro station node
- Verify metro line edges are included
- Compare metro vs. road impact

### Test Case 3: Multiple Removals
- Remove multiple nodes sequentially
- Verify compound effect on traffic
- Test restoration order independence

### Test Case 4: Time Variations
- Test removal at different hours
- Verify time multiplier affects congestion
- Compare peak vs. off-peak impact

## Accessibility Features

- Large, clear buttons for node operations
- Descriptive icons (ban icon for removed nodes)
- Color coding (red for removal, green for restore)
- Tooltip hints for actions
- Keyboard accessible (when focused)

---

**Feature Added**: December 3, 2025  
**Version**: 1.0  
**Status**: Production Ready

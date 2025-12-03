# Visual Guide: Node Removal & Traffic Impact Analysis

## 🗺️ Map Interface

### Before Node Removal
```
┌──────────────────────────────────────────────┐
│  Search Bar | Theme | Status | Device        │
├──────────────────────────────────────────────┤
│                                              │
│  Sidebar        │                   Map      │
│  ┌────────────┐ │  ◐     ◐ 🏥  ◐           │
│  │ Controls   │ │   ╱────╲  ╱────╲        │
│  │ Layers     │ │  ◐    ◐───◐  🚇         │
│  │ Actions    │ │   ╲────╱  ╲────╱        │
│  │            │ │  ◐  ◐ 🏫  ◐           │
│  │ Route      │ │   ╱────╲  ╱────╲        │
│  │ Planner    │ │                          │
│  │            │ │  ◐     ◐ 🌳  ◐           │
│  │ Road       │ │   ╱────╲  ╱────╲        │
│  │ Closure    │ │                          │
│  │            │ │                          │
│  │ Removed    │ │  Legend 🗺️               │
│  │ Nodes      │ │  • Blue: Roads           │
│  │ (empty)    │ │  • Red: Closed           │
│  │            │ │  • 🚇: Metro            │
│  │ Statistics │ │  • 🏥: Hospital         │
│  └────────────┘ │                          │
│                                              │
└──────────────────────────────────────────────┘
```

### After Node Removal
```
┌──────────────────────────────────────────────┐
│  Search Bar | Theme | Status | Device        │
├──────────────────────────────────────────────┤
│                                              │
│  Sidebar        │                   Map      │
│  ┌────────────┐ │  ◐     ◐ 🏥  ◐           │
│  │ Controls   │ │   ╱────╲  ╱╌╌╌╲  ← Pink │
│  │ Layers     │ │  ◐    ◐───◐  🚇    Dashed│
│  │ Actions    │ │   ╲─╌╌╱  ╲╌╌╌╱          │
│  │            │ │  ◐  X 🏫  ◐   (Removed) │
│  │ Route      │ │   ╱╌╌╌╲  ╱────╲        │
│  │ Planner    │ │                          │
│  │            │ │  ◐     ◐ 🌳  ◐           │
│  │ Road       │ │   ╱────╲  ╱────╲        │
│  │ Closure    │ │                          │
│  │            │ │  Legend 🗺️               │
│  │ Removed    │ │  • Blue: Open Roads      │
│  │ Nodes      │ │  • Pink ╌╌╌: Closed    │
│  │ ┌────────┐ │ │  • 🚇: Metro            │
│  │ │Node 156│ │ │  • 🏥: Hospital         │
│  │ │3 edges │ │ │                         │
│  │ │Hospital│ │ │  Node Info Panel →      │
│  │ │ [Rest] │ │ │                         │
│  │ └────────┘ │ │  Node 156               │
│  │ Statistics │ │  Removal Impact         │
│  │            │ │                         │
│  └────────────┘ │  Zone: North Quarter    │
│                                              │
└──────────────────────────────────────────────┘
```

---

## 🎛️ UI Components Breakdown

### Removed Nodes Panel (Sidebar)
```
┌─────────────────────────┐
│ 🚫 Removed Nodes        │
├─────────────────────────┤
│ Click on nodes to       │
│ remove/restore them     │
├─────────────────────────┤
│ ┌─────────────────────┐ │
│ │ Node 156            │ │
│ │ 3 edges closed      │ │
│ │ Hospital            │ │
│ │ [Restore] button    │ │
│ └─────────────────────┘ │
├─────────────────────────┤
│ ┌─────────────────────┐ │
│ │ Node 42             │ │
│ │ 8 edges closed      │ │
│ │ Metro Station       │ │
│ │ [Restore] button    │ │
│ └─────────────────────┘ │
└─────────────────────────┘
```

### Node Info Panel (Right Overlay)
```
┌─────────────────────────────────────────────┐
│ ✕  Node 156 - Removal Impact Analysis       │
├─────────────────────────────────────────────┤
│                                             │
│ Node Details                                │
│ ────────────────────────────────────────    │
│ Zone:           North Quarter               │
│ Population:     12,345                      │
│ Amenity:        Hospital                    │
│ Metro Station:  No                          │
│ Position:       (18.52, 73.87)              │
│                                             │
│ Traffic Impact                              │
│ ────────────────────────────────────────    │
│ Edges Closed:                    3          │
│ Mean Congestion (Closed):        62.3%      │
│ Max Congestion (Closed):         87.5%      │
│ Overall Mean Congestion:         41.2%      │
│ Overall Max Congestion:          91.0%      │
│                                             │
│ Transport Mode Impact                       │
│ ────────────────────────────────────────    │
│ Road Average:                    39.8%      │
│ Metro Average:                   52.1%      │
│                                             │
│ ┌────────────────────────────────────────┐ │
│ │ [🔄 Restore Node] (Green Button)      │ │
│ └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

---

## 🔄 User Interaction Flow Diagram

### Removing a Node
```
Start
  ↓
Click on node on map
  ↓
showNodeInfo() called
  ↓
Display node details panel
  ↓
User sees "Remove Node" button
  ↓
User clicks "Remove Node"
  ↓
removeNode(nodeId) called
  ↓
Loading indicator shown
  ↓
API: POST /api/analyze-node-removal
  ↓
Backend finds connected edges
  ↓
Backend runs GNN prediction
  ↓
Backend calculates impact stats
  ↓
API returns impact_analysis
  ↓
Frontend stores in state:
  - removedNodes.add(nodeId)
  - nodeImpactAnalysis[nodeId] = data
  ↓
Close all affected edges in UI
  ↓
Mark edges as closed in closedRoads Set
  ↓
Update edge visualization (pink, dashed)
  ↓
Call runPrediction() for network-wide stats
  ↓
showNodeRemovalImpact() displays results
  ↓
updateRemovedNodesList() updates sidebar
  ↓
Success toast notification
  ↓
End
```

### Restoring a Node
```
Start
  ↓
User sees removed node in sidebar
  ↓
User clicks "Restore" button
  ↓
restoreNode(nodeId) called
  ↓
Loading indicator shown
  ↓
Find all connected edges
  ↓
Remove from closedRoads Set
  ↓
Update edge visualization (blue, solid)
  ↓
Remove from state:
  - removedNodes.delete(nodeId)
  - delete nodeImpactAnalysis[nodeId]
  ↓
Call runPrediction() for updates
  ↓
updateRemovedNodesList() refreshes UI
  ↓
Success toast notification
  ↓
End
```

---

## 📊 Data Structure Visualization

### State Object
```javascript
state = {
    // Existing properties...
    closedRoads: Set {
        "0-1", "1-0", "42-156", "156-42", ...
    },
    
    // NEW: Node removal properties
    removedNodes: Set {
        156,    // Hospital
        42      // Metro Station
    },
    
    nodeImpactAnalysis: {
        156: {
            removed_node: 156,
            node_details: {
                id: 156,
                zone: "North Quarter",
                population: 12345,
                amenity: "hospital",
                x: 18.52,
                y: 73.87
            },
            closed_edges_count: 3,
            closed_edge_predictions: [0.623, 0.875, 0.567],
            mean_closed_edge_congestion: 0.688,
            max_closed_edge_congestion: 0.875,
            mean_congestion: 0.412,
            max_congestion: 0.91,
            road_mean: 0.398,
            metro_mean: 0.521
        },
        42: {
            // Similar structure for node 42
        }
    }
}
```

---

## 🌐 API Request/Response

### Request
```http
POST /api/analyze-node-removal HTTP/1.1
Content-Type: application/json

{
    "node_id": "156",
    "hour": 9
}
```

### Response (Success)
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
    "impact_analysis": {
        "removed_node": "156",
        "node_details": { ... },
        "closed_edges_count": 3,
        "closed_edge_predictions": [0.623, 0.875, 0.567],
        "mean_closed_edge_congestion": 0.688,
        "max_closed_edge_congestion": 0.875,
        "mean_congestion": 0.412,
        "max_congestion": 0.91,
        "road_mean": 0.398,
        "metro_mean": 0.521
    },
    "affected_edges": [
        "156-42", "42-156", "156-201"
    ],
    "predictions": [ ... ]
}
```

### Response (Error)
```http
HTTP/1.1 404 Not Found
Content-Type: application/json

{
    "error": "Node 999 not found"
}
```

---

## 🎨 Color Scheme

### Semantic Colors
```
Remove Node Action:     🔴 Red     (#e74c3c)
Restore Node Action:    🟢 Green   (#2ecc71)
Affected Edges:         🌸 Pink    (#ff69b4)
Closed Edge Pattern:    ╌╌╌ Dashed (10px, 6px)
Impact Data:            🔵 Cyan    (#00bcd4)
Critical Values:        🔴 Red     (Highlighted)
```

### UI States
```
Normal Edge:       Blue solid line, weight 2
Closed Edge:       Pink dashed line, weight 4
Metro Line:        Colored solid line, weight 5
Metro + Closed:    Same color but dashed

Normal Node:       Gray circle, radius 3
Removed Node:      Orange background in sidebar
Selected Node:     Info panel visible
```

---

## ⏱️ Timeline Visualization

### Typical User Session
```
T+0s:    Application loads
T+2s:    User clicks node (e.g., Hospital)
T+2.5s:  Info panel opens
T+3s:    User clicks "Remove Node"
T+3.1s:  Loading indicator starts
T+3.5s:  API processes request
T+3.7s:  Edges visualization updates
T+3.8s:  Impact analysis displays
T+3.9s:  Loading ends, toast shown
T+5s:    User reviews impact
T+7s:    User clicks "Restore"
T+7.1s:  Restoration processing
T+7.3s:  Visualization updated
T+7.4s:  Confirmation shown
```

---

## 🔍 Accessibility Features

### Keyboard Navigation
```
Tab:           Move between buttons
Enter:         Activate focused button
Esc:           Close info panel
Arrow Keys:    (for future keyboard navigation)
```

### Screen Reader
```
Button Text:   "Remove Node" / "Restore Node"
Icons:         Descriptive alt text
Statistics:    Semantic HTML structure
Contrast:      WCAG AA compliant (4.5:1)
```

### Mobile View
```
Sidebar:       Collapsible menu
Buttons:       Large touch targets (44x44px)
Panel:         Scrollable on small screens
Map:           Full width, pinch zoom
```

---

## 📋 Status Indicators

### Node Info Panel Status
```
🟢 Ready:       "Remove Node" button enabled
🔵 Processing:  "Loading..." indicator visible
🟠 Removed:     "Restore Node" button shown
🔴 Error:       Error message displayed
```

### Toast Notifications
```
✓ Success:     "Node X removed - Impact analysis complete"
✓ Success:     "Node X restored"
⚠ Warning:     "Node X is already removed"
✗ Error:       "Node removal analysis failed: [reason]"
ℹ Info:        "Analyzing traffic impact..."
```

---

## 🚀 Performance Indicators

### Response Time Targets
```
API Response:        100-500ms (depends on network)
UI Update:           <100ms
Loading Display:     Immediate
Total User Wait:     ~1 second average
```

### Memory Usage
```
Per Removed Node:    ~100 bytes
Typical Session:     <1MB total
Max Scalability:     Limited by browser, not code
```

---

## 🎓 Interaction Patterns

### Pattern 1: Quick Removal
```
Click Node → Remove Button → Review → Restore
Time: 3-5 seconds
```

### Pattern 2: Comparative Analysis
```
Remove Node A → Record Stats
Restore Node A
Remove Node B → Compare Stats
Restore Node B
```

### Pattern 3: Impact Understanding
```
Remove High-Degree Node → See Large Impact
Remove Low-Degree Node → See Small Impact
Compare Patterns
```

---

**Visual Documentation Complete** ✅  
**Last Updated**: December 3, 2025  
**Format**: ASCII Diagrams + HTML Descriptions

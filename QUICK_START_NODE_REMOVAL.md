# Quick Start: Node Removal & Traffic Impact Analysis

## 🚀 Getting Started in 5 Steps

### Step 1: Start the Application
```bash
cd "c:\Users\COMP\OneDrive\Desktop\EDI project\4\GNN---DIGITAL_CITY_SIMULATION"
python backend/app.py
```
- Open browser to `http://localhost:5000`
- Wait for graph to load (shows nodes and roads on map)

### Step 2: Locate a Node
**Option A - Search**:
- Use search bar at top: "Search locations..."
- Type node ID (e.g., "0") or amenity (e.g., "hospital")
- Click result to navigate

**Option B - Direct Click**:
- Click any node on the map
- Circles = regular nodes
- Emojis = amenities (🏥 hospital, 🚇 metro, etc.)

### Step 3: Open Node Info Panel
- Clicking a node shows info panel on the right
- Displays: Zone, Population, Amenity, Metro Status, Coordinates
- **"Remove Node" button** appears at the bottom

### Step 4: Remove the Node
- Click **"Remove Node"** button (red)
- System analyzes impact (loading spinner appears)
- Affected edges turn **pink with dashed lines**
- Impact analysis appears in the info panel

### Step 5: Review Impact Analysis
The info panel shows:

| Metric | Meaning |
|--------|---------|
| **Edges Closed** | Number of roads/metro lines affected |
| **Mean Congestion (Closed Edges)** | Average traffic on affected routes |
| **Max Congestion (Closed Edges)** | Worst-case traffic jam on affected routes |
| **Overall Mean Congestion** | Network-wide average impact |
| **Road/Metro Average** | Impact by transport mode |

---

## 💡 Practical Examples

### Example 1: Analyzing Metro Station Impact
```
1. Search for "metro_station"
2. Click on a metro station node (🚇)
3. Click "Remove Node"
4. See which metro lines and connected roads are affected
5. Check congestion predictions
6. Click "Restore" to revert
```

### Example 2: Comparing Different Hours
```
1. Adjust time slider to peak hours (8-9 AM or 5-7 PM)
2. Remove a high-traffic node
3. Note the congestion statistics
4. Restore the node
5. Change to off-peak hours (3 AM)
6. Remove same node
7. Compare statistics between peak and off-peak
```

### Example 3: Cascading Impact Analysis
```
1. Remove a major intersection (high-degree node)
2. Check the "Removed Nodes" panel
3. Note number of edges closed
4. Run prediction to see network-wide impact
5. Look at Road vs Metro breakdown
6. Remove another node to see compound effect
7. Analyze total impact
```

---

## 🎯 Key UI Elements

### Sidebar Components

**Removed Nodes Panel**:
```
┌─────────────────────┐
│ 🚫 Removed Nodes    │
├─────────────────────┤
│ Node 42             │
│ 5 edges closed      │
│ Metro Station       │
│ [Restore] button    │
├─────────────────────┤
│ Node 156            │
│ 3 edges closed      │
│ Hospital            │
│ [Restore] button    │
└─────────────────────┘
```

**Removed Nodes List**:
- Shows all currently removed nodes
- Displays affected edge count
- Shows amenity type for context
- One-click restoration

### Info Panel (Right Overlay)

**Node Details**:
```
Zone:        downtown
Population:  50,000
Amenity:     metro_station
Metro:       Yes
Position:    (18.52, 73.85)
```

**After Removal - Impact Analysis**:
```
Edges Closed:              8
Mean Congestion:           62.3%
Max Congestion:            87.5%
Overall Network Mean:      41.2%
Road Average:              39.8%
Metro Average:             52.1%
```

---

## 📊 Understanding the Metrics

### Congestion Percentage
- **0-20%** 🟢 Low: Smooth traffic
- **20-40%** 🟡 Medium: Moderate delays
- **40-60%** 🟠 High: Significant congestion
- **60-80%** 🔴 Very High: Severe delays
- **80-100%** ⚫ Critical: Complete congestion

### Edge Impact Colors
- **Blue lines** = Normal roads
- **Purple lines** = Metro lines
- **Pink dashed lines** = Closed/affected roads

### Node Impact Categories
- **Small nodes** (regular circles) = Regular intersections
- **Emoji markers** = Amenities (hospitals, schools, etc.)
- **Highlighted** = Currently selected

---

## 🔄 Restoration Options

### Individual Restore
- Click removed node in "Removed Nodes" panel
- Click **"Restore"** button
- Or click node on map → Click **"Restore Node"** button

### View Impact Before Restoring
- Info panel shows complete impact data
- Helps decide if restoration is necessary
- Can compare multiple removals

### Clear All
- Click **"Clear All"** button in Road Closure panel
- Removes all road closures AND reopens all node-removed edges
- Resets to baseline state

---

## ⚠️ Important Notes

### What Happens When You Remove a Node
1. ✓ All edges connected to the node are closed
2. ✓ Traffic is rerouted through alternate paths
3. ✓ Congestion increases on detour routes
4. ✓ GNN model predicts new congestion levels
5. ✓ Network statistics are recalculated

### What Does NOT Happen
1. ✗ The node itself is not deleted from the graph
2. ✗ Other nodes are not affected directly
3. ✗ No permanent changes to the network
4. ✗ Can always restore the node

### Limitations
- Can only remove one node at a time (current implementation)
- Node ID must exist in the graph
- Requires model to be loaded
- Analysis time depends on network size

---

## 🚀 Advanced Features

### Multi-Node Analysis (Sequential)
```
1. Remove Node A → Note statistics
2. Remove Node B → Compare to Node A
3. Remove Node C → Analyze compound effect
4. Restore all → See baseline
```

### Time-Based Impact
```
1. Set time to 9 AM (peak)
2. Remove critical node → High congestion
3. Set time to 3 AM (off-peak)
4. Same node removal → Lower congestion
5. Understand vulnerability at different times
```

### Transport Mode Analysis
```
1. Remove a metro station node
2. Check "Metro Average" vs "Road Average"
3. Identify which mode is more affected
4. Plan mitigation based on mode-specific impact
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Node not found" error | Verify node exists (check node count in status) |
| Button doesn't respond | Ensure backend is running (check API status) |
| No impact change | Refresh page, reload graph data |
| Edges don't close | Browser may need refresh, check console for errors |
| Slow analysis | Large graph takes longer, be patient |

---

## 📝 Tips & Tricks

1. **Identify Critical Nodes**: Remove nodes with high edge counts (hubs)
2. **Compare Modes**: Check road vs metro impact separately
3. **Peak vs Off-Peak**: Test same node at different times
4. **Chain Analysis**: Remove nodes that depend on each other
5. **Document Results**: Take notes on which nodes are most critical

---

## 🎓 Learning Outcomes

After using this feature, you'll understand:
- ✓ How node failures cascade through networks
- ✓ Traffic rerouting and congestion prediction
- ✓ Critical infrastructure identification
- ✓ Time-dependent impacts on transportation
- ✓ GNN model predictions in practice

---

## 📚 Related Features

- **Road Closure**: Manually close individual roads
- **Route Planner**: Find alternate routes between nodes
- **Prediction**: Run traffic analysis for current state
- **Analysis Page**: Export and compare scenarios
- **Time Slider**: Analyze different time periods

---

## ❓ FAQ

**Q: Can I remove multiple nodes at once?**  
A: Currently no, remove nodes sequentially. Multiple removals are shown in the sidebar.

**Q: Will the removal affect other users?**  
A: No, removals are stored client-side and don't persist between sessions.

**Q: Can I save my analysis?**  
A: Yes! Use "View Analysis" to see and export detailed impact data.

**Q: What if I remove a node with no edges?**  
A: The system will show 0 edges closed, minimal impact.

**Q: Does the GNN model update?**  
A: No, it's pre-trained. Predictions reflect learned patterns with given constraints.

---

**Version**: 1.0  
**Last Updated**: December 3, 2025  
**Status**: Ready to Use ✅

import networkx as nx
import matplotlib.pyplot as plt

print("🗺️  Loading Complex City Map...")
# Force return as a DiGraph to ensure compatibility, or let it auto-detect
try:
    G = nx.read_graphml("city_graph.graphml")
except Exception as e:
    print(f"Error loading graph: {e}")
    exit()

# 1. Extract GPS Coordinates
pos = {}
for node, data in G.nodes(data=True):
    # Handle potential missing coordinates safely
    if 'x' in data and 'y' in data:
        pos[node] = (float(data['x']), float(data['y']))
    else:
        # Fallback layout if coordinates are missing (shouldn't happen)
        print("Warning: GPS coordinates missing. Using spring layout.")
        pos = nx.spring_layout(G)
        break

# 2. Define Node Colors (Zones)
node_colors = []
for node, data in G.nodes(data=True):
    zone = data.get('zone', 'suburbs')
    if zone == 'industrial':
        node_colors.append('#ff3333')    # Bright Red
    elif zone == 'residential':
        node_colors.append('#33ff33')    # Bright Green
    elif zone == 'downtown':
        node_colors.append('#3333ff')    # Bright Blue
    else:
        node_colors.append('#444444')    # Dark Gray (Suburbs)

# 3. Define Edge Colors (Road Hierarchy)
edge_colors = []
edge_widths = []

# --- FIX: REMOVED 'keys=True' to prevent TypeError ---
for u, v, data in G.edges(data=True):
    if data.get('highway') == 'primary':
        edge_colors.append('#ffa500')    # Orange (Highways)
        edge_widths.append(2.0)          # Thick
    else:
        edge_colors.append('#222222')    # Very Dark Gray (Streets)
        edge_widths.append(0.5)          # Thin

# 4. Plotting
print(f"   Nodes: {len(node_colors)} | Edges: {len(edge_colors)}")
plt.figure(figsize=(12, 12), facecolor='black')

# Draw the graph
nx.draw_networkx_nodes(G, pos, node_size=15, node_color=node_colors, alpha=0.8, node_shape='o')
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrows=False)

# Highlight hospitals if present
hospital_nodes = [
    node for node, data in G.nodes(data=True)
    if data.get('amenity') == 'hospital' and node in pos
]
if hospital_nodes:
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=hospital_nodes,
        node_size=200,
        node_color='#ffeb3b',
        edgecolors='black',
        linewidths=0.8,
        node_shape='*',
        label='Hospitals'
    )

# Highlight green zones
green_nodes = [
    node for node, data in G.nodes(data=True)
    if data.get('green_zone') and node in pos
]
if green_nodes:
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=green_nodes,
        node_size=60,
        node_color='#00e676',
        edgecolors='white',
        linewidths=0.5,
        alpha=0.9,
        node_shape='s',
        label='Green Zones'
    )

# Decorate
plt.title("Synthetic Pune: Organic Structure (Voronoi Generation)", color='white', fontsize=16)
plt.text(min(x for x,y in pos.values()), min(y for x,y in pos.values()), 
         "Highways (Orange) | Industrial (Red) | Downtown (Blue) | Hospitals (Yellow Stars) | Green Zones (Neon Squares)", color='white', fontsize=10)
plt.axis('off')

print("✨ Displaying map...")
plt.show()
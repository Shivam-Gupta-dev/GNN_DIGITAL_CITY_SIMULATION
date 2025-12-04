"""Test CTM initialization and step"""
import networkx as nx
from ctm_traffic_simulation import CTMTrafficSimulator, CTMConfig

print("Loading graph...")
graph = nx.read_graphml('city_graph.graphml')
if not isinstance(graph, nx.MultiDiGraph):
    graph = nx.MultiDiGraph(graph)

print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

print("\nInitializing CTM...")
try:
    config = CTMConfig()
    ctm = CTMTrafficSimulator(graph, config)
    print("✅ CTM initialized successfully!")
    
    print("\nTrying 1 step...")
    ctm.step(save_snapshot=True, show_progress=True)
    print("✅ Step 1 completed!")
    
    print("\nTrying 5 more steps...")
    for i in range(5):
        ctm.step(save_snapshot=True, show_progress=True)
    print("✅ All 5 steps completed!")
    
    print("\nGetting statistics...")
    stats = ctm.get_statistics()
    print(f"Stats: {stats}")
    print("✅ Statistics retrieved!")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

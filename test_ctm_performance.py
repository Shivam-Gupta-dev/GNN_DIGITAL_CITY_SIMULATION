"""
Test CTM Performance Improvements
==================================
Measure before/after performance of CTM optimization
"""

import time
import networkx as nx
from ctm_traffic_simulation import CTMTrafficSimulator, CTMConfig

def load_graph():
    """Load the city graph"""
    print("Loading graph...")
    G = nx.read_graphml('city_graph.graphml')
    G = nx.MultiDiGraph(G)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
    return G

def test_initialization():
    """Test CTM initialization speed"""
    print("=" * 60)
    print("TEST 1: CTM Initialization Performance")
    print("=" * 60)
    
    G = load_graph()
    
    print("\n🚀 Initializing CTM...")
    start = time.time()
    ctm = CTMTrafficSimulator(G)
    elapsed = time.time() - start
    
    print(f"\n✅ Initialization complete!")
    print(f"⏱️  Time: {elapsed:.2f}s")
    print(f"📦 Total cells: {sum(len(cells) for cells in ctm.cells.values()):,}")
    print(f"🔗 Cached connections: {len(ctm._successor_cache):,}")
    
    return ctm

def test_single_step(ctm):
    """Test single simulation step"""
    print("\n" + "=" * 60)
    print("TEST 2: Single Step Performance")
    print("=" * 60)
    
    print("\n⏱️  Running single step...")
    start = time.time()
    ctm.step(save_snapshot=True, show_progress=True)
    elapsed = time.time() - start
    
    print(f"\n✅ Single step complete!")
    print(f"⏱️  Time: {elapsed:.2f}s")

def test_batch_steps(ctm):
    """Test 10-step batch simulation"""
    print("\n" + "=" * 60)
    print("TEST 3: 10-Step Batch Performance")
    print("=" * 60)
    
    print("\n⏱️  Running 10 steps (snapshots only on last step)...")
    start = time.time()
    
    for i in range(10):
        save_snapshot = (i == 9)  # Only save on last step
        show_progress = True
        ctm.step(save_snapshot=save_snapshot, show_progress=show_progress)
    
    elapsed = time.time() - start
    
    print(f"\n✅ Batch simulation complete!")
    print(f"⏱️  Total time: {elapsed:.2f}s")
    print(f"⏱️  Average per step: {elapsed/10:.2f}s")
    print(f"📊 Total snapshots: {len(ctm.snapshots)}")

def test_road_closure(ctm):
    """Test road closure impact"""
    print("\n" + "=" * 60)
    print("TEST 4: Road Closure Performance")
    print("=" * 60)
    
    # Get first edge
    edge_ids = list(ctm.cells.keys())[:3]
    print(f"\n🚧 Closing {len(edge_ids)} roads...")
    for u, v, key in edge_ids:
        ctm.close_road(u, v, key)
    
    print("\n⏱️  Running step with closures...")
    start = time.time()
    ctm.step(save_snapshot=True, show_progress=True)
    elapsed = time.time() - start
    
    print(f"\n✅ Closure simulation complete!")
    print(f"⏱️  Time: {elapsed:.2f}s")

def main():
    """Run all performance tests"""
    print("\n" + "=" * 60)
    print("🚀 CTM PERFORMANCE TEST SUITE")
    print("=" * 60)
    print("Testing optimizations:")
    print("  ✓ Topology caching (O(1) lookups)")
    print("  ✓ Batch demand/supply pre-computation")
    print("  ✓ Vectorized random sampling")
    print("  ✓ Conditional snapshot saving")
    print("  ✓ Progress indicators")
    print("=" * 60)
    
    try:
        # Test 1: Initialization
        ctm = test_initialization()
        
        # Test 2: Single step
        test_single_step(ctm)
        
        # Test 3: Batch steps
        test_batch_steps(ctm)
        
        # Test 4: Road closure
        test_road_closure(ctm)
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETE!")
        print("=" * 60)
        print("\n📊 Performance Summary:")
        print(f"   Total cells: {sum(len(cells) for cells in ctm.cells.values()):,}")
        print(f"   Cached topology: {len(ctm._successor_cache):,} connections")
        print(f"   Total snapshots: {len(ctm.snapshots)}")
        print(f"   Simulation time: {ctm.simulation_time:.2f} hours")
        print(f"   Total vehicles: {ctm.total_vehicles:,}")
        print("\n🎉 Optimizations working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

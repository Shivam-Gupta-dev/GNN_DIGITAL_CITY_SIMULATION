#!/usr/bin/env python3
"""
Quick CTM Performance Test
===========================
Run this to verify the CTM optimizations are working.
Should complete in under 30 seconds.
"""

import time
import networkx as nx
from ctm_traffic_simulation import CTMTrafficSimulator, CTMConfig

print("="*60)
print("CTM QUICK PERFORMANCE TEST")
print("="*60)

# Load graph
print("\n1. Loading graph...")
start = time.time()
G = nx.read_graphml('city_graph.graphml')
if not isinstance(G, nx.MultiDiGraph):
    G = nx.MultiDiGraph(G)
print(f"   ✅ Loaded in {time.time()-start:.2f}s")
print(f"   📊 {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Initialize CTM
print("\n2. Initializing CTM (optimized)...")
start = time.time()
config = CTMConfig(
    cell_length_km=1.0,          # Optimized default
    fast_mode=False               # Build caches upfront
)
ctm = CTMTrafficSimulator(G, config)
init_time = time.time() - start
print(f"   ✅ Initialized in {init_time:.2f}s")
print(f"   📦 Total cells: {sum(len(cells) for cells in ctm.cells.values()):,}")

# Run single step
print("\n3. Running single simulation step...")
start = time.time()
ctm.step(save_snapshot=True, show_progress=False)
step_time = time.time() - start
print(f"   ✅ Step completed in {step_time:.2f}s")

# Run 10 steps
print("\n4. Running 10 simulation steps...")
start = time.time()
for i in range(10):
    save_snapshot = (i == 9)  # Only save last one
    ctm.step(save_snapshot=save_snapshot, show_progress=False)
batch_time = time.time() - start
print(f"   ✅ 10 steps completed in {batch_time:.2f}s")
print(f"   ⚡ Average: {batch_time/10:.2f}s per step")

# Get statistics
print("\n5. Getting statistics...")
start = time.time()
stats = ctm.get_statistics()
stats_time = time.time() - start
print(f"   ✅ Stats retrieved in {stats_time:.2f}s")
print(f"   📊 Congestion: {stats.get('average_congestion', 0):.1%}")
print(f"   🚗 Vehicles: {stats.get('total_vehicles', 0):,}")

# Performance summary
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)
print(f"Initialization:  {init_time:6.2f}s  {'✅ GOOD' if init_time < 30 else '⚠️ SLOW'}")
print(f"Single step:     {step_time:6.2f}s  {'✅ GOOD' if step_time < 2 else '⚠️ SLOW'}")
print(f"10-step batch:   {batch_time:6.2f}s  {'✅ GOOD' if batch_time < 15 else '⚠️ SLOW'}")
print(f"Statistics:      {stats_time:6.2f}s  {'✅ GOOD' if stats_time < 1 else '⚠️ SLOW'}")
print("="*60)

total_time = init_time + step_time + batch_time + stats_time
if total_time < 45:
    print("\n🎉 EXCELLENT! CTM optimizations are working perfectly!")
elif total_time < 90:
    print("\n✅ GOOD! Performance is acceptable.")
else:
    print("\n⚠️ WARNING! Performance is slower than expected.")
    print("   Check if graph is too large or system is under load.")

print(f"\nTotal test time: {total_time:.2f}s")

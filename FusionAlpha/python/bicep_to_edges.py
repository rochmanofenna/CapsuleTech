#!/usr/bin/env python3
"""
Convert BICEP trajectories to edge weights for Fusion Alpha graphs
Reads BICEP parquet files and computes transition statistics
"""

import numpy as np
import pandas as pd
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse


def discretize_state(state: np.ndarray, bounds: np.ndarray, n_bins: int = 20) -> int:
    """
    Discretize continuous state to grid index
    
    Args:
        state: continuous state vector
        bounds: (2, d) min/max bounds per dimension
        n_bins: discretization resolution per dimension
    
    Returns:
        grid_index: flattened grid index
    """
    # Normalize to [0, 1] per dimension
    normalized = (state - bounds[0]) / (bounds[1] - bounds[0] + 1e-8)
    normalized = np.clip(normalized, 0, 0.999)
    
    # Convert to grid indices
    indices = (normalized * n_bins).astype(int)
    
    # Flatten to single index (for simplicity, just use first 2 dims)
    if len(indices) >= 2:
        return indices[0] * n_bins + indices[1]
    else:
        return int(indices[0])


def compute_transition_stats(trajectories: pd.DataFrame, 
                           n_bins: int = 20,
                           dt: float = 0.01) -> Dict:
    """
    Compute transition statistics from BICEP trajectories
    
    Returns dict with:
        - transitions: {(i,j): count}
        - variances: {(i,j): variance}
        - state_bounds: (2, d) array
    """
    # Extract state sequences
    states = []
    for traj in trajectories['state']:
        if isinstance(traj, list):
            states.extend(traj)
    
    states = np.array(states)
    
    # Compute bounds
    state_bounds = np.array([
        states.min(axis=0),
        states.max(axis=0)
    ])
    
    # Count transitions
    transitions = defaultdict(int)
    transition_times = defaultdict(list)
    
    for _, row in trajectories.iterrows():
        traj_states = np.array(row['state'])
        
        for t in range(len(traj_states) - 1):
            # Discretize states
            i = discretize_state(traj_states[t], state_bounds, n_bins)
            j = discretize_state(traj_states[t+1], state_bounds, n_bins)
            
            transitions[(i, j)] += 1
            transition_times[(i, j)].append(dt)  # Fixed dt for now
    
    # Compute variances
    variances = {}
    for (i, j), times in transition_times.items():
        if len(times) > 1:
            variances[(i, j)] = np.var(times)
        else:
            variances[(i, j)] = 0.1  # Default variance
    
    return {
        'transitions': dict(transitions),
        'variances': dict(variances),
        'state_bounds': state_bounds.tolist(),
        'n_bins': n_bins,
        'n_states': n_bins * n_bins,  # Assuming 2D for now
    }


def transitions_to_edge_weights(stats: Dict, 
                               temperature: float = 1.0) -> List[Tuple[int, int, float]]:
    """
    Convert transition statistics to edge weights
    
    Uses: w_ij = count_ij / max_count * exp(-var_ij / temperature)
    """
    transitions = stats['transitions']
    variances = stats['variances']
    
    if not transitions:
        return []
    
    max_count = max(transitions.values())
    edges = []
    
    for (i, j), count in transitions.items():
        # Skip self-loops for cleaner graphs
        if i == j:
            continue
            
        # Count-based weight
        count_weight = count / max_count
        
        # Variance penalty
        var = variances.get((i, j), 0.1)
        var_weight = np.exp(-var / temperature)
        
        # Combined weight
        weight = count_weight * var_weight
        
        # Store as (i, j, weight)
        edges.append((int(i), int(j), float(weight)))
    
    return edges


def create_grid_nodes(n_bins: int, bounds: np.ndarray) -> np.ndarray:
    """Create node positions for discretized grid"""
    nodes = []
    
    for i in range(n_bins):
        for j in range(n_bins):
            # Map back to continuous space
            x = bounds[0, 0] + (i + 0.5) * (bounds[1, 0] - bounds[0, 0]) / n_bins
            y = bounds[0, 1] + (j + 0.5) * (bounds[1, 1] - bounds[0, 1]) / n_bins if bounds.shape[1] > 1 else 0.0
            nodes.append([x, y])
    
    return np.array(nodes, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert BICEP trajectories to graph edges")
    parser.add_argument("--bicep-parquet", required=True, help="Path to BICEP trajectory parquet")
    parser.add_argument("--output", default="bicep_graph.json", help="Output JSON file")
    parser.add_argument("--n-bins", type=int, default=20, help="Discretization resolution")
    parser.add_argument("--temperature", type=float, default=1.0, help="Variance temperature")
    
    args = parser.parse_args()
    
    print(f"Loading BICEP trajectories from {args.bicep_parquet}")
    df = pd.read_parquet(args.bicep_parquet)
    print(f"Loaded {len(df)} trajectories")
    
    # Compute transition statistics
    print(f"Computing transition statistics (n_bins={args.n_bins})...")
    stats = compute_transition_stats(df, n_bins=args.n_bins)
    
    n_transitions = len(stats['transitions'])
    print(f"Found {n_transitions} unique transitions")
    
    # Convert to edge weights
    print("Converting to edge weights...")
    edges = transitions_to_edge_weights(stats, temperature=args.temperature)
    print(f"Created {len(edges)} edges")
    
    # Create node positions
    bounds = np.array(stats['state_bounds'])
    nodes = create_grid_nodes(args.n_bins, bounds)
    
    # Save results
    output = {
        'nodes': nodes.tolist(),
        'edges': edges,
        'stats': stats,
        'n_nodes': len(nodes),
        'n_edges': len(edges),
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved graph to {args.output}")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Edge weight range: [{min(e[2] for e in edges):.3f}, {max(e[2] for e in edges):.3f}]")
    
    # Print example edges
    print("\nExample edges (top 10 by weight):")
    sorted_edges = sorted(edges, key=lambda e: e[2], reverse=True)[:10]
    for i, j, w in sorted_edges:
        print(f"  {i} → {j}: {w:.3f}")


if __name__ == "__main__":
    main()
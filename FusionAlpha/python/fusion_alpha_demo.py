#!/usr/bin/env python3
"""
Fusion Alpha Python Demo
Shows how to use the Python bindings for committor planning
"""

import numpy as np
import sys
import os

# Add the built library to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'target', 'release'))

try:
    import fusion_alpha as fa
except ImportError as e:
    print(f"Failed to import fusion_alpha: {e}")
    print("Make sure to build with: cargo build --release -p fusion-bindings")
    sys.exit(1)

def demo_humanoid_maze():
    """Demo humanoid maze navigation"""
    print("=== Humanoid Maze Demo ===")
    
    # Define observation
    obs = {
        'x': 1.0,
        'y': 1.0, 
        'angle': 0.0,
        'vx': 0.0,
        'vy': 0.0,
    }
    
    # Define goal
    goal = {
        'x': 8.0,
        'y': 8.0,
        'radius': 1.0,
    }
    
    # Define maze (10x10 grid)
    maze = {
        'width': 10,
        'height': 10,
        'cell_size': 0.5,
        'walls': [False] * 100,  # No walls for simplicity
    }
    
    # Build graph
    graph, current_node, goal_node = fa.build_humanoid_graph_py(obs, goal, maze)
    print(f"Built graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    print(f"Current node: {current_node}, Goal node: {goal_node}")
    
    # Set up priors
    q0 = np.full(graph.num_nodes(), np.nan, dtype=np.float32)
    eta = np.zeros(graph.num_nodes(), dtype=np.float32)
    
    # Goal boundary condition
    q0[goal_node] = 1.0
    eta[goal_node] = 1e9
    
    # ENN prior for current state
    q0[current_node] = 0.6  # Mock ENN prediction
    eta[current_node] = 0.7  # Mock confidence
    
    priors = fa.PyPriors(q0, eta)
    
    # Propagation config
    config = fa.PyPropConfig(t_max=50, eps=1e-4, use_parallel=True)
    
    # Mock ENN state for severity scaling
    enn_state = fa.PyFusionState(q_prior_enn=0.6, severity=0.8, bicep_confidence=0.7)
    t_steps = enn_state.propagation_steps(config.t_max)
    
    print(f"Propagation steps: {t_steps} (severity-scaled)")
    
    # Propagate committor values
    q_values = fa.propagate_committor_py(graph, priors, config, t_steps)
    
    print(f"Committor values computed")
    print(f"Current node q: {q_values[current_node]:.3f}")
    print(f"Goal node q: {q_values[goal_node]:.3f}")
    
    # Pick next action
    next_node = fa.pick_next_node_py(graph, q_values, current_node)
    
    if next_node is not None:
        print(f"Selected next node: {next_node} (q = {q_values[next_node]:.3f})")
        
        # Show neighbors
        neighbors = graph.neighbors(current_node)
        print("Neighbors:")
        for neighbor_id, weight in neighbors:
            print(f"  Node {neighbor_id}: q = {q_values[neighbor_id]:.3f}, weight = {weight:.2f}")
    else:
        print("No valid next node found")
    
    return q_values

def demo_ant_soccer():
    """Demo ant soccer planning"""
    print("\n=== Ant Soccer Demo ===")
    
    obs = {
        'ant_x': 3.0,
        'ant_y': 4.0,
        'ant_angle': 0.0,
        'ball_x': 4.0,
        'ball_y': 4.0,
        'ball_vx': 0.0,
        'ball_vy': 0.0,
    }
    
    goal = {
        'x': 12.0,
        'y': 4.0,
        'width': 2.0,
        'direction': 0.0,  # Right goal
    }
    
    field = {
        'width': 12.0,
        'height': 8.0,
        'cell_size': 0.4,
    }
    
    try:
        graph, current_node, goal_node = fa.build_soccer_graph_py(obs, goal, field)
        print(f"Built soccer graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
        print(f"Current ball node: {current_node}, Goal node: {goal_node}")
        
        # Quick propagation test
        q0 = np.full(graph.num_nodes(), np.nan, dtype=np.float32)
        eta = np.zeros(graph.num_nodes(), dtype=np.float32)
        q0[goal_node] = 1.0
        eta[goal_node] = 1e9
        
        priors = fa.PyPriors(q0, eta)
        config = fa.PyPropConfig(t_max=30, eps=1e-4, use_parallel=True)
        
        q_values = fa.propagate_committor_py(graph, priors, config, 20)
        print(f"Soccer committor values computed, current ball q: {q_values[current_node]:.3f}")
        
    except Exception as e:
        print(f"Soccer demo failed: {e}")

def demo_puzzle():
    """Demo puzzle solving"""
    print("\n=== Puzzle Demo ===")
    
    # Simple puzzle: all off -> all on
    initial_config = 0  # All lights off
    goal_config = 0xFFFFF  # All lights on (20 bits)
    depth = 4
    
    try:
        graph, current_node, goal_node = fa.build_puzzle_graph_py(initial_config, goal_config, depth)
        print(f"Built puzzle graph: {graph.num_nodes()} nodes, {graph.num_edges()} edges")
        print(f"Current config node: {current_node}, Goal node: {goal_node}")
        
        # Set up priors
        q0 = np.full(graph.num_nodes(), np.nan, dtype=np.float32)
        eta = np.zeros(graph.num_nodes(), dtype=np.float32)
        q0[goal_node] = 1.0
        eta[goal_node] = 1e9
        
        # Mock ENN prediction for puzzle solving
        q0[current_node] = 0.3  # Low initial confidence
        eta[current_node] = 0.5
        
        priors = fa.PyPriors(q0, eta)
        config = fa.PyPropConfig(t_max=40, eps=1e-4, use_parallel=True)
        
        q_values = fa.propagate_committor_py(graph, priors, config, 30)
        print(f"Puzzle committor values computed, current q: {q_values[current_node]:.3f}")
        
        # Show some neighbors (button press options)
        neighbors = graph.neighbors(current_node)
        print(f"Available button presses ({len(neighbors)} options):")
        for i, (neighbor_id, weight) in enumerate(neighbors[:5]):  # Show first 5
            print(f"  Option {i+1}: node {neighbor_id}, q = {q_values[neighbor_id]:.3f}")
            
    except Exception as e:
        print(f"Puzzle demo failed: {e}")

def demo_risk_sensitive():
    """Demo risk-sensitive propagation"""
    print("\n=== Risk-Sensitive Demo ===")
    
    # Create simple chain graph: 0 -- 1 -- 2 (goal)
    nodes = np.array([
        [0.0, 0.0],  # Node 0
        [1.0, 0.0],  # Node 1  
        [2.0, 0.0],  # Node 2 (goal)
    ], dtype=np.float32)
    
    edges = np.array([
        [0, 1, 1.0],  # 0 -> 1
        [1, 0, 1.0],  # 1 -> 0
        [1, 2, 1.0],  # 1 -> 2
        [2, 1, 1.0],  # 2 -> 1
    ], dtype=np.float32)
    
    graph = fa.PyGraph(nodes, edges)
    
    # Set up priors
    q0 = np.full(3, np.nan, dtype=np.float32)
    eta = np.zeros(3, dtype=np.float32)
    q0[2] = 1.0  # Goal
    eta[2] = 1e9
    
    priors = fa.PyPriors(q0, eta)
    config = fa.PyPropConfig(t_max=50, eps=1e-4, use_parallel=True)
    
    # Compare different risk attitudes
    alphas = [2.0, 0.0, -2.0]  # Pessimistic, Neutral, Optimistic
    labels = ['Pessimistic', 'Regular', 'Optimistic']
    
    print("Risk-sensitive comparison:")
    for alpha, label in zip(alphas, labels):
        if alpha == 0.0:
            q_vals = fa.propagate_committor_py(graph, priors, config, 30)
        else:
            q_vals = fa.propagate_risk_sensitive_py(graph, priors, alpha, config, 30)
        
        print(f"  {label:12} (α={alpha:4.1f}): q[0]={q_vals[0]:.3f}, q[1]={q_vals[1]:.3f}, q[2]={q_vals[2]:.3f}")

def main():
    print("Fusion Alpha Python Bindings Demo\n")
    
    try:
        # Run all demos
        demo_humanoid_maze()
        demo_ant_soccer()
        demo_puzzle()
        demo_risk_sensitive()
        
        print("\n✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
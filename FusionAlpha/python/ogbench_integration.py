#!/usr/bin/env python3
"""
OGBench Integration for Fusion Alpha
Connects Fusion Alpha committor planning to OGBench environments
"""

import numpy as np
import sys
import os
from typing import Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod

# Add fusion_alpha to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'target', 'release'))

try:
    import fusion_alpha as fa
except ImportError as e:
    print(f"Warning: Could not import fusion_alpha: {e}")
    print("Build with: cargo build --release -p fusion-bindings")

class FusionAlphaPlanner(ABC):
    """Base class for Fusion Alpha planning in OGBench environments"""
    
    def __init__(self, 
                 severity_scale: float = 1.0,
                 base_confidence: float = 0.8,
                 t_max: int = 50,
                 eps: float = 1e-4):
        self.severity_scale = severity_scale
        self.base_confidence = base_confidence
        self.t_max = t_max
        self.eps = eps
        
    @abstractmethod
    def build_graph(self, obs: Dict, goal: Dict) -> Tuple[Any, int, int]:
        """Build task graph from observation and goal"""
        pass
    
    @abstractmethod
    def obs_to_dict(self, obs) -> Dict:
        """Convert environment observation to dict format"""
        pass
    
    @abstractmethod 
    def action_from_node(self, obs, current_node_id: int, next_node_id: int) -> Any:
        """Convert selected node to environment action"""
        pass
    
    def plan_action(self, 
                   obs,
                   goal: Dict,
                   enn_q_prior: float = 0.5,
                   enn_severity: float = 0.3,
                   bicep_confidence: Optional[float] = None) -> Any:
        """Main planning interface"""
        
        # Convert observation
        obs_dict = self.obs_to_dict(obs)
        
        # Build graph
        graph, current_node, goal_node = self.build_graph(obs_dict, goal)
        
        # Set up priors
        q0 = np.full(graph.num_nodes(), np.nan, dtype=np.float32)
        eta = np.zeros(graph.num_nodes(), dtype=np.float32)
        
        # Goal boundary
        q0[goal_node] = 1.0
        eta[goal_node] = 1e9
        
        # ENN prior
        q0[current_node] = enn_q_prior
        eta[current_node] = bicep_confidence or self.base_confidence
        
        priors = fa.PyPriors(q0, eta)
        
        # Propagation config
        config = fa.PyPropConfig(self.t_max, self.eps, use_parallel=True)
        
        # Severity-scaled propagation
        enn_state = fa.PyFusionState(enn_q_prior, enn_severity, self.base_confidence)
        t_steps = enn_state.propagation_steps(self.t_max)
        
        # Propagate committor
        q_values = fa.propagate_committor_py(graph, priors, config, t_steps)
        
        # Select next node
        next_node = fa.pick_next_node_py(graph, q_values, current_node)
        
        if next_node is None:
            return None  # No valid action
            
        # Convert to environment action
        return self.action_from_node(obs, current_node, next_node)

class HumanoidMazePlanner(FusionAlphaPlanner):
    """Fusion Alpha planner for HumanoidMaze environments"""
    
    def __init__(self, maze_config: Dict, **kwargs):
        super().__init__(**kwargs)
        self.maze_config = maze_config
        
    def build_graph(self, obs: Dict, goal: Dict) -> Tuple[Any, int, int]:
        return fa.build_humanoid_graph_py(obs, goal, self.maze_config)
        
    def obs_to_dict(self, obs) -> Dict:
        """Convert HumanoidMaze observation to dict"""
        # Assuming obs is a dict-like or has attributes
        if hasattr(obs, 'x'):  # Object with attributes
            return {
                'x': float(obs.x),
                'y': float(obs.y), 
                'angle': getattr(obs, 'angle', 0.0),
                'vx': getattr(obs, 'vx', 0.0),
                'vy': getattr(obs, 'vy', 0.0),
            }
        elif isinstance(obs, dict):  # Already a dict
            return obs
        elif hasattr(obs, '__getitem__'):  # Array-like
            return {
                'x': float(obs[0]),
                'y': float(obs[1]),
                'angle': float(obs[2]) if len(obs) > 2 else 0.0,
                'vx': float(obs[3]) if len(obs) > 3 else 0.0,
                'vy': float(obs[4]) if len(obs) > 4 else 0.0,
            }
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")
    
    def action_from_node(self, obs, current_node_id: int, next_node_id: int) -> np.ndarray:
        """Convert waypoint to velocity action"""
        # This would interface with the actual graph nodes
        # For now, return a simple forward action
        return np.array([1.0, 0.0], dtype=np.float32)  # [forward, angular]

class AntSoccerPlanner(FusionAlphaPlanner):
    """Fusion Alpha planner for AntSoccer environments"""
    
    def __init__(self, field_config: Dict, **kwargs):
        super().__init__(**kwargs)
        self.field_config = field_config
        
    def build_graph(self, obs: Dict, goal: Dict) -> Tuple[Any, int, int]:
        return fa.build_soccer_graph_py(obs, goal, self.field_config)
        
    def obs_to_dict(self, obs) -> Dict:
        """Convert AntSoccer observation to dict"""
        if hasattr(obs, 'ant_x'):
            return {
                'ant_x': float(obs.ant_x),
                'ant_y': float(obs.ant_y),
                'ant_angle': getattr(obs, 'ant_angle', 0.0),
                'ball_x': float(obs.ball_x),
                'ball_y': float(obs.ball_y),
                'ball_vx': getattr(obs, 'ball_vx', 0.0),
                'ball_vy': getattr(obs, 'ball_vy', 0.0),
            }
        elif isinstance(obs, dict):
            return obs
        else:
            # Parse from array (common format)
            return {
                'ant_x': float(obs[0]),
                'ant_y': float(obs[1]),
                'ant_angle': float(obs[2]) if len(obs) > 2 else 0.0,
                'ball_x': float(obs[3]),
                'ball_y': float(obs[4]),
                'ball_vx': float(obs[5]) if len(obs) > 5 else 0.0,
                'ball_vy': float(obs[6]) if len(obs) > 6 else 0.0,
            }
    
    def action_from_node(self, obs, current_node_id: int, next_node_id: int) -> np.ndarray:
        """Convert ball target to ant torques"""
        # Mock ant action - in practice would use learned controller
        n_joints = 8
        return np.random.uniform(-0.3, 0.3, n_joints).astype(np.float32)

class PuzzlePlanner(FusionAlphaPlanner):
    """Fusion Alpha planner for Puzzle-4x5 environments"""
    
    def __init__(self, depth: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        
    def build_graph(self, obs: Dict, goal: Dict) -> Tuple[Any, int, int]:
        current_config = obs['config']
        goal_config = goal['config']
        return fa.build_puzzle_graph_py(current_config, goal_config, self.depth)
        
    def obs_to_dict(self, obs) -> Dict:
        """Convert Puzzle observation to dict"""
        if hasattr(obs, 'config'):
            return {'config': int(obs.config)}
        elif isinstance(obs, dict):
            return obs
        elif isinstance(obs, int):
            return {'config': obs}
        else:
            # Assume it's a binary array of lights
            config = 0
            for i, light in enumerate(obs[:20]):  # First 20 elements
                if light > 0.5:
                    config |= (1 << i)
            return {'config': config}
    
    def action_from_node(self, obs, current_node_id: int, next_node_id: int) -> int:
        """Convert node transition to button press"""
        # Would need to query the graph to find which button press
        # connects current_node to next_node
        # For now, return random button
        return np.random.randint(0, 20)

def create_ogbench_planner(env_name: str, env_config: Dict) -> FusionAlphaPlanner:
    """Factory function to create appropriate planner for OGBench environment"""
    
    if 'humanoid' in env_name.lower() or 'maze' in env_name.lower():
        maze_config = {
            'width': env_config.get('width', 20),
            'height': env_config.get('height', 20), 
            'cell_size': env_config.get('cell_size', 0.5),
            'walls': env_config.get('walls', [False] * 400),  # Default no walls
        }
        return HumanoidMazePlanner(maze_config)
        
    elif 'ant' in env_name.lower() and 'soccer' in env_name.lower():
        field_config = {
            'width': env_config.get('width', 12.0),
            'height': env_config.get('height', 8.0),
            'cell_size': env_config.get('cell_size', 0.4),
        }
        return AntSoccerPlanner(field_config)
        
    elif 'puzzle' in env_name.lower():
        return PuzzlePlanner(depth=env_config.get('depth', 6))
        
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

# Example usage function
def demo_ogbench_integration():
    """Demo integration with mock OGBench environments"""
    
    # Mock HumanoidMaze
    print("=== OGBench Integration Demo ===")
    
    # Create planners
    humanoid_config = {
        'width': 10,
        'height': 10,
        'cell_size': 0.5,
        'walls': [False] * 100,
    }
    
    planner = create_ogbench_planner('HumanoidMaze-v1', humanoid_config)
    
    # Mock observation and goal
    obs = {'x': 1.0, 'y': 1.0, 'angle': 0.0}
    goal = {'x': 8.0, 'y': 8.0, 'radius': 1.0}
    
    # Plan action (mock ENN/BICEP inputs)
    action = planner.plan_action(
        obs=obs,
        goal=goal,
        enn_q_prior=0.7,  # ENN's q prediction
        enn_severity=0.4,  # ENN contradiction level
        bicep_confidence=0.8,  # BICEP path reliability
    )
    
    print(f"Planned action: {action}")
    print("✅ OGBench integration demo completed")

if __name__ == "__main__":
    demo_ogbench_integration()
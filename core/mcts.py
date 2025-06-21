import math
import random
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
from gymnasium_env.env.rubiks_cube import RubiksCubeEnv


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree."""
    
    def __init__(self, state: np.ndarray, parent: Optional['MCTSNode'] = None, action: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = list(range(12))  # All possible actions
        
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried from this node."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self, env: RubiksCubeEnv) -> bool:
        """Check if this node represents a terminal state."""
        env.cube_state = self.state.reshape(6, 9)
        return env._is_solved()
    
    def select_child(self, exploration_constant: float = math.sqrt(2)) -> 'MCTSNode':
        """Select child using UCB1 formula."""
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                ucb_score = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
                ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def add_child(self, action: int, state: np.ndarray) -> 'MCTSNode':
        """Add a child node for the given action and state."""
        child = MCTSNode(state, parent=self, action=action)
        self.children[action] = child
        self.untried_actions.remove(action)
        return child
    
    def update(self, reward: float):
        """Update this node with simulation result."""
        self.visits += 1
        self.value += reward
    
    def backpropagate(self, reward: float):
        """Backpropagate the reward up the tree."""
        self.update(reward)
        if self.parent:
            self.parent.backpropagate(reward)


class MCTS:
    """Monte Carlo Tree Search implementation for Rubik's Cube."""
    
    def __init__(self, env: RubiksCubeEnv, exploration_constant: float = math.sqrt(2)):
        self.env = env
        self.exploration_constant = exploration_constant
        self.root = None
    
    def search(self, initial_state: np.ndarray, num_simulations: int = 1000) -> int:
        """Run MCTS and return the best action."""
        self.root = MCTSNode(initial_state)
        
        for _ in range(num_simulations):
            # Selection and Expansion
            node = self._select_and_expand(self.root)
            
            # Simulation
            reward = self._simulate(node)
            
            # Backpropagation
            node.backpropagate(reward)
        
        # Return action with highest visit count
        return self._get_best_action()
    
    def _select_and_expand(self, node: MCTSNode) -> MCTSNode:
        """Select and expand a node in the tree."""
        # Traverse down the tree using UCB1
        while not node.is_terminal(self.env) and node.is_fully_expanded():
            node = node.select_child(self.exploration_constant)
        
        # If we found a terminal node, return it
        if node.is_terminal(self.env):
            return node
        
        # Otherwise, expand by adding a new child
        action = random.choice(node.untried_actions)
        
        # Apply action to get new state
        self.env.cube_state = node.state.reshape(6, 9)
        self.env._apply_action(action)
        new_state = self.env.cube_state.flatten()
        
        return node.add_child(action, new_state)
    
    def _simulate(self, node: MCTSNode) -> float:
        """Run a random simulation from the given node."""
        # Set environment to node's state
        self.env.cube_state = node.state.reshape(6, 9)
        
        # Check if already solved
        if self.env._is_solved():
            return 1.0
        
        # Run random simulation
        max_depth = 50  # Limit simulation depth
        for step in range(max_depth):
            action = random.randint(0, 11)
            self.env._apply_action(action)
            
            if self.env._is_solved():
                # Reward is higher for shorter solutions
                return 1.0 / (step + 1)
        
        # If not solved, give partial reward based on progress
        return self._evaluate_state()
    
    def _evaluate_state(self) -> float:
        """Evaluate the current state of the cube."""
        solved_faces = 0
        total_correct_squares = 0
        
        for face in range(6):
            face_state = self.env.cube_state[face]
            center_color = face_state[4]  # Center square (position 4)
            
            # Count correct squares on this face
            correct_squares = sum(1 for color in face_state if color == center_color)
            total_correct_squares += correct_squares
            
            # Check if face is completely solved
            if correct_squares == 9:
                solved_faces += 1
        
        # Reward based on progress
        face_bonus = solved_faces * 0.1
        square_bonus = (total_correct_squares / 54) * 0.1
        
        return face_bonus + square_bonus
    
    def _get_best_action(self) -> int:
        """Get the action with the highest visit count."""
        if not self.root.children:
            return random.randint(0, 11)
        
        best_action = max(self.root.children.keys(), 
                         key=lambda action: self.root.children[action].visits)
        return best_action


def solve_cube_with_mcts(scrambles: int = 5, max_moves: int = 100, 
                        simulations_per_move: int = 1000) -> Tuple[bool, List[int], int]:
    """
    Solve a Rubik's cube using MCTS.
    
    Args:
        scrambles: Number of random moves to scramble the cube
        max_moves: Maximum number of moves allowed to solve
        simulations_per_move: Number of MCTS simulations per move
    
    Returns:
        (solved, solution_moves, total_moves)
    """
    env = RubiksCubeEnv(scrambles=scrambles, max_steps=max_moves)
    mcts = MCTS(env)
    
    # Reset and scramble the cube
    initial_state, _ = env.reset()
    
    print(f"Starting MCTS solver with {scrambles} scrambles...")
    print(f"Initial state - Solved faces: {env._get_info()['solved_faces']}")
    
    solution_moves = []
    
    for move_count in range(max_moves):
        if env._is_solved():
            print(f"Cube solved in {move_count} moves!")
            return True, solution_moves, move_count
        
        # Get current state
        current_state = env.cube_state.flatten()
        
        # Run MCTS to find best action
        start_time = time.time()
        best_action = mcts.search(current_state, simulations_per_move)
        search_time = time.time() - start_time
        
        # Apply the action
        env._apply_action(best_action)
        solution_moves.append(best_action)
        
        # Print progress
        info = env._get_info()
        print(f"Move {move_count + 1}: Action {best_action}, "
              f"Solved faces: {info['solved_faces']}, "
              f"Search time: {search_time:.2f}s")
        
        # Create new MCTS instance for next move (fresh tree)
        mcts = MCTS(env)
    
    print(f"Failed to solve cube within {max_moves} moves")
    return False, solution_moves, max_moves


def main():
    """Main function to test MCTS solver."""
    print("Testing MCTS Rubik's Cube Solver")
    print("=" * 40)
    
    # Test with different difficulty levels
    test_cases = [
        (3, 50, 500),   # Easy: 3 scrambles, 50 max moves, 500 simulations
        (5, 100, 1000),  # Medium: 5 scrambles, 100 max moves, 1000 simulations
        (7, 150, 1500),  # Hard: 7 scrambles, 150 max moves, 1500 simulations
    ]
    
    for i, (scrambles, max_moves, simulations) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {scrambles} scrambles")
        print("-" * 30)
        
        solved, moves, total_moves = solve_cube_with_mcts(
            scrambles=scrambles,
            max_moves=max_moves,
            simulations_per_move=simulations
        )
        
        if solved:
            print(f"✓ Success! Solved in {total_moves} moves")
            print(f"Solution: {moves}")
        else:
            print(f"✗ Failed to solve within {max_moves} moves")
        
        print(f"Total moves attempted: {total_moves}")


if __name__ == "__main__":
    main()
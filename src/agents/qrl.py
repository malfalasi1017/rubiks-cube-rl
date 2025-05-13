import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QRLAgent(nn.Module):
    def __init__(self, qnn, action_space):
        super().__init__()
        self.qnn = qnn
        self.action_space = action_space
        # Store the number of actions for easier access
        self.num_actions = action_space.n

    def encode_state(self, state):
        """
        Encode the 54-dimensional Rubik's Cube state into a 6-dimensional feature vector.
        
        Args:
            state: The 54-dimensional Rubik's Cube state (6 faces * 9 squares)
            
        Returns:
            A 6-dimensional feature vector suitable for the quantum circuit
        """
        # Convert to numpy array if not already
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        state = np.array(state)
        
        # Reshape to 6 faces x 9 squares if flattened
        if state.shape == (54,):
            state = state.reshape(6, 9)
            
        # Feature 1: Proportion of solved faces (0-1)
        solved_faces = 0
        for face in range(6):
            first_color = state[face][0]
            if all(color == first_color for color in state[face]):
                solved_faces += 1
        feature_1 = solved_faces / 6.0
        
        # Feature 2-6: Color distribution features
        # For each face, calculate how uniform the colors are
        color_uniformity = []
        for face in range(6):
            # Count occurrences of most common color
            unique, counts = np.unique(state[face], return_counts=True)
            max_count = np.max(counts)
            uniformity = max_count / 9.0  # 9 squares per face
            color_uniformity.append(uniformity)
            
        # Combine features into a 6D vector and normalize to [0,1] range
        features = np.array([feature_1] + color_uniformity[:5])  # Take only 5 face uniformity to make 6 total features
        
        return features

    def forward(self, state):
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = np.array(state)
            
        # Encode the state to 6 dimensions for the quantum circuit
        encoded_state = self.encode_state(state)
        
        # Convert to tensor for QNN
        encoded_state_tensor = torch.tensor(encoded_state, dtype=torch.float32)
        
        # Process through QNN
        qnn_output = self.qnn(encoded_state_tensor)
        
        # Map the QNN output to 12 actions
        # If QNN output is less than 12 dimensions, pad with zeros
        if len(qnn_output.shape) == 0:  # If it's a scalar, convert to a batch of 1
            qnn_output = qnn_output.unsqueeze(0)
            
        # Expand/map to the number of actions if needed
        if qnn_output.shape[0] < self.num_actions:
            logits = torch.zeros(self.num_actions)
            # Copy values for available actions
            for i in range(min(qnn_output.shape[0], self.num_actions)):
                logits[i] = qnn_output[i]
                
            # For remaining actions, set to small negative values (to make them less likely)
            for i in range(qnn_output.shape[0], self.num_actions):
                logits[i] = -0.1
        else:
            # If we have exactly the right number or more, take the first self.num_actions
            logits = qnn_output[:self.num_actions]
            
        return logits

    def select_action(self, state):
        with torch.no_grad():
            logits = self.forward(state)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action
    
    def update(self, state, action, rewards):
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = np.array(state)
            
        # Encode the state to 6 dimensions for the quantum circuit
        encoded_state = self.encode_state(state)
        
        # Convert to tensor for QNN
        encoded_state_tensor = torch.tensor(encoded_state, dtype=torch.float32)
        
        # Process through QNN
        qnn_output = self.qnn(encoded_state_tensor)
        
        # Calculate loss (mean squared error) for the selected action
        target = torch.zeros(self.num_actions)
        target[action] = rewards
        
        loss = nn.MSELoss()(qnn_output, target)
        
        return loss

        
        
    
    
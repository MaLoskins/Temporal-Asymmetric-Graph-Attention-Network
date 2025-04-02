"""
Memory bank utility for TAGAN.

This module provides a memory bank for storing node states across time steps,
handling node appearance, disappearance, and reappearance.
"""

import torch
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union, Any


class NodeMemoryBank:
    """
    Memory bank for storing node states across time steps.
    
    This class maintains a dictionary mapping node IDs to their hidden states
    and keeps track of node activity over time.
    
    Attributes:
        hidden_dim (int): Hidden state dimension
        decay_factor (float): Decay factor for inactive nodes
        max_inactivity (int): Maximum number of inactive steps before pruning
        device (torch.device): Device to store tensors on
        node_states (dict): Dictionary mapping node IDs to hidden states
        inactivity_counter (dict): Dictionary mapping node IDs to inactivity counters
    """
    
    def __init__(
        self,
        hidden_dim: int,
        decay_factor: float = 0.8,
        max_inactivity: int = 5,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the node memory bank.
        
        Args:
            hidden_dim: Hidden state dimension
            decay_factor: Decay factor for inactive nodes (default: 0.8)
            max_inactivity: Maximum number of inactive steps before pruning (default: 5)
            device: Device to store tensors on (default: None, uses CPU)
        """
        self.hidden_dim = hidden_dim
        self.decay_factor = decay_factor
        self.max_inactivity = max_inactivity
        self.device = device if device is not None else torch.device('cpu')
        
        # Dictionary mapping node IDs to hidden states
        self.node_states = {}
        
        # Dictionary mapping node IDs to inactivity counters
        self.inactivity_counter = {}
    
    def update(
        self,
        node_ids: List[int],
        states: torch.Tensor
    ):
        """
        Update node states in the memory bank.
        
        Args:
            node_ids: List of node IDs
            states: Tensor of node states [num_nodes, hidden_dim]
        """
        # Ensure states tensor is on the correct device
        states = states.to(self.device)
        
        # Increment inactivity counter for all nodes
        for node_id in self.inactivity_counter:
            self.inactivity_counter[node_id] += 1
        
        # Update states and reset inactivity counter for active nodes
        for i, node_id in enumerate(node_ids):
            self.node_states[node_id] = states[i].clone()
            self.inactivity_counter[node_id] = 0
        
        # Apply decay to inactive nodes
        for node_id in self.node_states:
            if node_id not in node_ids:
                self.node_states[node_id] = self.node_states[node_id] * self.decay_factor
        
        # Prune nodes that have been inactive for too long
        for node_id in list(self.inactivity_counter.keys()):
            if self.inactivity_counter[node_id] > self.max_inactivity:
                # Remove from memory
                del self.node_states[node_id]
                del self.inactivity_counter[node_id]
    
    def get_state(self, node_id: int) -> Optional[torch.Tensor]:
        """
        Get the state for a specific node.
        
        Args:
            node_id: Node ID to retrieve
            
        Returns:
            Node state tensor if node exists, None otherwise
        """
        return self.node_states.get(node_id, None)
    
    def get_states(self, node_ids: List[int]) -> torch.Tensor:
        """
        Get states for a list of nodes.
        
        Args:
            node_ids: List of node IDs to retrieve
            
        Returns:
            Tensor of node states [num_nodes, hidden_dim]
        """
        states = []
        
        for node_id in node_ids:
            if node_id in self.node_states:
                # Use existing state
                states.append(self.node_states[node_id])
            else:
                # Initialize new state with zeros
                states.append(torch.zeros(self.hidden_dim, device=self.device))
                
                # Add to memory
                self.node_states[node_id] = states[-1].clone()
                self.inactivity_counter[node_id] = 0
        
        return torch.stack(states)
    
    def get_active_nodes(self) -> List[int]:
        """
        Get list of currently active nodes.
        
        Returns:
            List of active node IDs
        """
        return list(self.node_states.keys())
    
    def decay_all(self):
        """Apply decay to all nodes in the memory bank."""
        for node_id in self.node_states:
            self.node_states[node_id] = self.node_states[node_id] * self.decay_factor
    
    def reset(self):
        """Reset the memory bank, clearing all states."""
        self.node_states = {}
        self.inactivity_counter = {}
    
    def save(self, filepath: str):
        """
        Save the memory bank to a file.
        
        Args:
            filepath: Path to save the memory bank
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Convert tensors to CPU for saving
        node_states_cpu = {
            node_id: state.cpu() for node_id, state in self.node_states.items()
        }
        
        # Create state dictionary
        state_dict = {
            'hidden_dim': self.hidden_dim,
            'decay_factor': self.decay_factor,
            'max_inactivity': self.max_inactivity,
            'node_states': node_states_cpu,
            'inactivity_counter': self.inactivity_counter.copy()
        }
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f)
    
    def load(self, filepath: str):
        """
        Load the memory bank from a file.
        
        Args:
            filepath: Path to load the memory bank from
        """
        # Load from file
        with open(filepath, 'rb') as f:
            state_dict = pickle.load(f)
        
        # Update attributes
        self.hidden_dim = state_dict['hidden_dim']
        self.decay_factor = state_dict['decay_factor']
        self.max_inactivity = state_dict['max_inactivity']
        
        # Load node states and move to device
        self.node_states = {
            node_id: state.to(self.device) 
            for node_id, state in state_dict['node_states'].items()
        }
        
        # Load inactivity counter
        self.inactivity_counter = state_dict['inactivity_counter']
    
    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None):
        """
        Create a new memory bank from a saved file.
        
        Args:
            filepath: Path to load the memory bank from
            device: Device to store tensors on (default: None, uses CPU)
            
        Returns:
            New NodeMemoryBank instance
        """
        # Load from file
        with open(filepath, 'rb') as f:
            state_dict = pickle.load(f)
        
        # Create new instance
        memory_bank = cls(
            hidden_dim=state_dict['hidden_dim'],
            decay_factor=state_dict['decay_factor'],
            max_inactivity=state_dict['max_inactivity'],
            device=device
        )
        
        # Load node states and move to device
        memory_bank.node_states = {
            node_id: state.to(device) if device is not None else state
            for node_id, state in state_dict['node_states'].items()
        }
        
        # Load inactivity counter
        memory_bank.inactivity_counter = state_dict['inactivity_counter']
        
        return memory_bank
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory bank.
        
        Returns:
            Dictionary of memory bank statistics
        """
        num_nodes = len(self.node_states)
        avg_inactivity = (
            sum(self.inactivity_counter.values()) / num_nodes 
            if num_nodes > 0 else 0
        )
        
        return {
            'num_nodes': num_nodes,
            'avg_inactivity': avg_inactivity,
            'max_inactivity_limit': self.max_inactivity,
            'decay_factor': self.decay_factor,
            'hidden_dim': self.hidden_dim
        }
    
    def __repr__(self) -> str:
        """String representation of the memory bank."""
        return (f"NodeMemoryBank(hidden_dim={self.hidden_dim}, "
                f"decay_factor={self.decay_factor}, "
                f"max_inactivity={self.max_inactivity}, "
                f"active_nodes={len(self.node_states)})")


class TemporalMemoryBank:
    """
    Temporal memory bank for storing node states at each time step.
    
    This class extends NodeMemoryBank to maintain a history of node states
    over time, allowing for advanced temporal operations.
    
    Attributes:
        hidden_dim (int): Hidden state dimension
        max_history (int): Maximum number of time steps to store
        device (torch.device): Device to store tensors on
        decay_factor (float): Decay factor for inactive nodes
        node_history (dict): Dictionary mapping node IDs to lists of states
        time_index (int): Current time index
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_history: int = 10,
        decay_factor: float = 0.8,
        max_inactivity: int = 5,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the temporal memory bank.
        
        Args:
            hidden_dim: Hidden state dimension
            max_history: Maximum number of time steps to store (default: 10)
            decay_factor: Decay factor for inactive nodes (default: 0.8)
            max_inactivity: Maximum number of inactive steps before pruning (default: 5)
            device: Device to store tensors on (default: None, uses CPU)
        """
        self.hidden_dim = hidden_dim
        self.max_history = max_history
        self.decay_factor = decay_factor
        self.max_inactivity = max_inactivity
        self.device = device if device is not None else torch.device('cpu')
        
        # Dictionary mapping node IDs to lists of states (one per time step)
        self.node_history = {}
        
        # Dictionary mapping node IDs to inactivity counters
        self.inactivity_counter = {}
        
        # Current time index
        self.time_index = 0
    
    def update(
        self,
        node_ids: List[int],
        states: torch.Tensor,
        increment_time: bool = True
    ):
        """
        Update node states in the memory bank.
        
        Args:
            node_ids: List of node IDs
            states: Tensor of node states [num_nodes, hidden_dim]
            increment_time: Whether to increment the time index (default: True)
        """
        # Ensure states tensor is on the correct device
        states = states.to(self.device)
        
        # Increment inactivity counter for all nodes
        for node_id in self.inactivity_counter:
            self.inactivity_counter[node_id] += 1
        
        # Update states and reset inactivity counter for active nodes
        for i, node_id in enumerate(node_ids):
            state = states[i].clone()
            
            if node_id not in self.node_history:
                # Initialize with zeros for past time steps
                self.node_history[node_id] = [torch.zeros(self.hidden_dim, device=self.device)] * self.time_index
                
                # Add current state
                self.node_history[node_id].append(state)
            else:
                # Add padding if there are missing time steps
                missing_steps = self.time_index - len(self.node_history[node_id])
                
                if missing_steps > 0:
                    # Use last known state with decay
                    last_state = self.node_history[node_id][-1]
                    
                    for _ in range(missing_steps):
                        last_state = last_state * self.decay_factor
                        self.node_history[node_id].append(last_state.clone())
                
                # Add current state
                self.node_history[node_id].append(state)
            
            # Truncate history if it exceeds max_history
            if len(self.node_history[node_id]) > self.max_history:
                self.node_history[node_id] = self.node_history[node_id][-self.max_history:]
            
            # Reset inactivity counter
            self.inactivity_counter[node_id] = 0
        
        # Prune nodes that have been inactive for too long
        for node_id in list(self.inactivity_counter.keys()):
            if self.inactivity_counter[node_id] > self.max_inactivity:
                # Remove from memory
                if node_id in self.node_history:
                    del self.node_history[node_id]
                
                del self.inactivity_counter[node_id]
        
        # Increment time index
        if increment_time:
            self.time_index += 1
    
    def get_state(
        self,
        node_id: int,
        time_offset: int = 0
    ) -> Optional[torch.Tensor]:
        """
        Get the state for a specific node at a specific time offset.
        
        Args:
            node_id: Node ID to retrieve
            time_offset: Time offset relative to current time (default: 0)
                      Negative values indicate past states
            
        Returns:
            Node state tensor if node exists at that time, None otherwise
        """
        if node_id not in self.node_history:
            return None
        
        # Calculate absolute time index
        time_idx = self.time_index + time_offset
        
        # Convert to relative index in node history
        rel_idx = time_idx - (self.time_index - len(self.node_history[node_id]) + 1)
        
        if 0 <= rel_idx < len(self.node_history[node_id]):
            return self.node_history[node_id][rel_idx]
        else:
            return None
    
    def get_history(
        self,
        node_id: int,
        window_size: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Get the state history for a specific node.
        
        Args:
            node_id: Node ID to retrieve
            window_size: Number of past states to retrieve (default: None, all available)
            
        Returns:
            List of node state tensors
        """
        if node_id not in self.node_history:
            return []
        
        if window_size is None:
            return self.node_history[node_id]
        else:
            return self.node_history[node_id][-window_size:]
    
    def get_current_states(self, node_ids: List[int]) -> torch.Tensor:
        """
        Get current states for a list of nodes.
        
        Args:
            node_ids: List of node IDs to retrieve
            
        Returns:
            Tensor of node states [num_nodes, hidden_dim]
        """
        states = []
        
        for node_id in node_ids:
            if node_id in self.node_history and len(self.node_history[node_id]) > 0:
                # Use existing state
                states.append(self.node_history[node_id][-1])
            else:
                # Initialize new state with zeros
                states.append(torch.zeros(self.hidden_dim, device=self.device))
                
                # Add to memory
                self.node_history[node_id] = [states[-1].clone()]
                self.inactivity_counter[node_id] = 0
        
        return torch.stack(states)
    
    def get_interpolated_state(
        self,
        node_id: int,
        time_offset: float
    ) -> Optional[torch.Tensor]:
        """
        Get interpolated state for a specific node at a fractional time offset.
        
        Args:
            node_id: Node ID to retrieve
            time_offset: Time offset relative to current time (can be fractional)
            
        Returns:
            Interpolated node state tensor if node exists, None otherwise
        """
        if node_id not in self.node_history:
            return None
        
        # Calculate integer and fractional parts
        int_offset = int(time_offset)
        frac_offset = time_offset - int_offset
        
        # Get states at integer offsets
        state1 = self.get_state(node_id, int_offset)
        state2 = self.get_state(node_id, int_offset + 1)
        
        if state1 is None or state2 is None:
            return state1 if state1 is not None else state2
        
        # Interpolate between states
        return state1 * (1 - frac_offset) + state2 * frac_offset
    
    def get_active_nodes(self) -> List[int]:
        """
        Get list of currently active nodes.
        
        Returns:
            List of active node IDs
        """
        return list(self.node_history.keys())
    
    def reset(self):
        """Reset the memory bank, clearing all states."""
        self.node_history = {}
        self.inactivity_counter = {}
        self.time_index = 0
    
    def save(self, filepath: str):
        """
        Save the memory bank to a file.
        
        Args:
            filepath: Path to save the memory bank
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Convert tensors to CPU for saving
        node_history_cpu = {
            node_id: [state.cpu() for state in states]
            for node_id, states in self.node_history.items()
        }
        
        # Create state dictionary
        state_dict = {
            'hidden_dim': self.hidden_dim,
            'max_history': self.max_history,
            'decay_factor': self.decay_factor,
            'max_inactivity': self.max_inactivity,
            'node_history': node_history_cpu,
            'inactivity_counter': self.inactivity_counter.copy(),
            'time_index': self.time_index
        }
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f)
    
    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None):
        """
        Create a new temporal memory bank from a saved file.
        
        Args:
            filepath: Path to load the memory bank from
            device: Device to store tensors on (default: None, uses CPU)
            
        Returns:
            New TemporalMemoryBank instance
        """
        # Load from file
        with open(filepath, 'rb') as f:
            state_dict = pickle.load(f)
        
        # Create new instance
        memory_bank = cls(
            hidden_dim=state_dict['hidden_dim'],
            max_history=state_dict['max_history'],
            decay_factor=state_dict['decay_factor'],
            max_inactivity=state_dict['max_inactivity'],
            device=device
        )
        
        # Load node history and move to device
        memory_bank.node_history = {
            node_id: [
                state.to(device) if device is not None else state
                for state in states
            ]
            for node_id, states in state_dict['node_history'].items()
        }
        
        # Load inactivity counter and time index
        memory_bank.inactivity_counter = state_dict['inactivity_counter']
        memory_bank.time_index = state_dict['time_index']
        
        return memory_bank
    
    def __repr__(self) -> str:
        """String representation of the temporal memory bank."""
        return (f"TemporalMemoryBank(hidden_dim={self.hidden_dim}, "
                f"max_history={self.max_history}, "
                f"decay_factor={self.decay_factor}, "
                f"max_inactivity={self.max_inactivity}, "
                f"time_index={self.time_index}, "
                f"active_nodes={len(self.node_history)})")
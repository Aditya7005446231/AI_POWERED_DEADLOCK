import numpy as np
from typing import Tuple, Dict, List, Union

class BankersAlgorithm:
    def __init__(self):
        """Initialize Banker's Algorithm with empty state"""
        self.allocation: np.ndarray = None
        self.max_demand: np.ndarray = None
        self.available: np.ndarray = None
        self.n_processes: int = 0
        self.n_resources: int = 0

    def set_data(self, allocation: List[List[int]], max_demand: List[List[int]], available: List[int]) -> None:
        """Set the matrices for the algorithm with validation"""
        allocation = np.array(allocation)
        max_demand = np.array(max_demand)
        available = np.array(available)
        
        if allocation.shape != max_demand.shape:
            raise ValueError("Allocation and Max Demand matrices must have the same shape")
        if len(available.shape) != 1:
            raise ValueError("Available resources must be a 1D array")
        if allocation.shape[1] != available.shape[0]:
            raise ValueError("Number of resources in allocation must match available resources")
        if np.any(allocation > max_demand):
            raise ValueError("Allocation cannot exceed maximum demand for any resource")
        
        self.allocation = allocation
        self.max_demand = max_demand
        self.available = available
        self.n_processes = allocation.shape[0]
        self.n_resources = available.shape[0]

    def check_safety(self) -> Dict[str, Union[bool, List[int], List[np.ndarray], np.ndarray]]:
        """Check if current state is safe and return detailed results"""
        work = self.available.copy()
        finish = [False] * self.n_processes
        safe_sequence = []
        need = self.max_demand - self.allocation
        work_history = [work.copy()]
        
        while len(safe_sequence) < self.n_processes:
            found = False
            for i in range(self.n_processes):
                if not finish[i] and all(need[i] <= work):
                    work += self.allocation[i]
                    finish[i] = True
                    safe_sequence.append(i)
                    found = True
                    work_history.append(work.copy())
                    break
            
            if not found:
                break
        
        return {
            'is_safe': all(finish),
            'sequence': safe_sequence,
            'work_history': work_history,
            'need_matrix': need
        }

    def request_resources(self, process_id: int, request: List[int]) -> Tuple[bool, str]:
        """Handle resource request using Banker's algorithm"""
        request = np.array(request)
        
        if process_id < 0 or process_id >= self.n_processes:
            raise ValueError("Invalid process ID")
        if request.shape != (self.n_resources,):
            raise ValueError("Request must match number of resources")
        
        need = self.max_demand[process_id] - self.allocation[process_id]
        if np.any(request > need):
            return False, "Request exceeds process's maximum claim"
        if np.any(request > self.available):
            return False, "Insufficient resources available"
        
        # Try the allocation temporarily
        self.available -= request
        self.allocation[process_id] += request
        
        # Check safety
        is_safe, _ = self.check_safety()
        
        # Rollback if unsafe
        if not is_safe:
            self.available += request
            self.allocation[process_id] -= request
            return False, "Request would lead to unsafe state"
        
        return True, "Request granted"

    def get_need_matrix(self) -> np.ndarray:
        """Return the need matrix (max - allocation)"""
        return self.max_demand - self.allocation

    def reset(self) -> None:
        """Reset the algorithm state"""
        self.allocation = None
        self.max_demand = None
        self.available = None
        self.n_processes = 0
        self.n_resources = 0

    def format_sequence(self, sequence: List[int]) -> str:
        """Format the sequence as P0 → P1 → P2"""
        return ' → '.join([f'P{pid}' for pid in sequence])

    def print_state(self) -> None:
        """Print current state in readable format"""
        print("Current State:")
        print(f"Processes: {self.n_processes}, Resources: {self.n_resources}")
        print("\nAllocation Matrix:")
        print(self.allocation)
        print("\nMax Demand Matrix:")
        print(self.max_demand)
        print("\nAvailable Resources:")
        print(self.available)
        print("\nNeed Matrix:")
        print(self.get_need_matrix())
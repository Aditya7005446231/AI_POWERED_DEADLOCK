import numpy as np

class BankersAlgorithm:
    def __init__(self):
        self.allocation = None
        self.max_demand = None
        self.available = None
        self.n_processes = 0
        self.n_resources = 0

    def set_data(self, allocation, max_demand, available):
        """Set the matrices for the algorithm"""
        self.allocation = np.array(allocation)
        self.max_demand = np.array(max_demand)
        self.available = np.array(available)
        self.n_processes = len(allocation)
        self.n_resources = len(available)

    def check_safety(self):
        """Implement Banker's safety algorithm"""
        work = self.available.copy()
        finish = [False] * self.n_processes
        safe_sequence = []
        need = self.max_demand - self.allocation

        while len(safe_sequence) < self.n_processes:
            found = False
            for i in range(self.n_processes):
                if not finish[i] and all(need[i] <= work):
                    work += self.allocation[i]
                    finish[i] = True
                    safe_sequence.append(i)
                    found = True
                    break  # Move to next iteration after finding one

            if not found:
                break  # Deadlock detected

        return all(finish), safe_sequence

    def format_sequence(self, sequence):
        """Format the sequence as P0 → P1 → P2"""
        return ' → '.join([f'P{pid}' for pid in sequence])
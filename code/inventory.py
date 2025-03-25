
import simpy
from sympy.stats.sampling.sample_numpy import numpy

import numpy as np
import random

# Parameters
INITIAL_INVENTORY = 30  # Initial stock level
s = 20  # Reorder point
S = 50  # Restock level
MEAN_DEMAND_ARRIVAL = 2  # Mean time between demand (Exponential distribution)
MEAN_LEAD_TIME = 1  # Mean lead time for replenishment (Exponential distribution)
SIM_TIME = 100  # Total simulation time


class InventorySystem:
    def __init__(self, env):
        self.env = env
        self.inventory = INITIAL_INVENTORY  # Current inventory level
        self.orders = 0  # Number of outstanding orders
        self.action = env.process(self.run())  # Start simulation process

    def run(self):
        while True:
            # Wait for the next demand event
            yield self.env.timeout(random.expovariate(1 / MEAN_DEMAND_ARRIVAL))

            # Reduce inventory by 1 per demand
            self.inventory -= 1
            print(f"[{self.env.now:.2f}] Demand arrives. Inventory: {self.inventory}")

            # Check if we need to reorder
            if self.inventory <= s and self.orders == 0:
                self.reorder()

    def reorder(self):
        order_quantity = S - self.inventory  # Replenish to level S
        self.orders += order_quantity
        print(f"[{self.env.now:.2f}] Reordering {order_quantity} units")

        # Wait for lead time (exponentially distributed)
        yield self.env.timeout(random.expovariate(1 / MEAN_LEAD_TIME))

        # Receive the order
        self.inventory += order_quantity
        self.orders = 0
        print(f"[{self.env.now:.2f}] Order received. Inventory: {self.inventory}")


# Initialize simulation
env = simpy.Environment()
inventory_system = InventorySystem(env)

# Run the simulation
env.run(until=SIM_TIME)

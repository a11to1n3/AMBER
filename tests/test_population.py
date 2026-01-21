import unittest
import polars as pl
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ambr.population import Population

class TestPopulation(unittest.TestCase):
    def test_add_agent(self):
        pop = Population(schema={'wealth': pl.Int64})
        pop.add_agent(1, wealth=100)
        self.assertEqual(pop.size, 1)
        self.assertEqual(pop.get_agent_value(1, 'wealth'), 100)

    def test_batch_update_by_ids(self):
        pop = Population(schema={'wealth': pl.Int64})
        pop.batch_add_agents(3, wealth=0) # IDs 0, 1, 2
        
        # Update IDs 0 and 2
        pop.batch_update_by_ids([0, 2], {'wealth': [10, 20]})
        
        self.assertEqual(pop.get_agent_value(0, 'wealth'), 10)
        self.assertEqual(pop.get_agent_value(1, 'wealth'), 0)
        self.assertEqual(pop.get_agent_value(2, 'wealth'), 20)

    def test_batch_context(self):
        pop = Population(schema={'wealth': pl.Int64})
        pop.batch_add_agents(2, wealth=0)
        
        with pop.create_batch_context() as batch:
            batch.add_update(0, 'wealth', 50)
            batch.add_update(1, 'wealth', 100)
            
        self.assertEqual(pop.get_agent_value(0, 'wealth'), 50)
        self.assertEqual(pop.get_agent_value(1, 'wealth'), 100)

if __name__ == '__main__':
    unittest.main()

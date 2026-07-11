"""Tests for ambr.population module."""

import pytest
import polars as pl
from unittest.mock import Mock, patch

from ambr.population import Population, BatchUpdateContext
from polars.exceptions import ColumnNotFoundError


class TestPopulation:
    """Test cases for Population class."""

    def test_add_agent(self):
        pop = Population(schema={'wealth': pl.Int64})
        pop.add_agent(1, wealth=100)
        assert pop.size == 1
        assert pop.get_agent_value(1, 'wealth') == 100

    def test_add_agent_without_optional_columns(self):
        pop = Population(schema={'wealth': pl.Int64})
        pop.add_agent(1)  # wealth is optional
        assert pop.size == 1
        # Optional columns should be null
        assert pop.get_agent_value(1, 'wealth') is None

    def test_add_agent_with_extra_column(self):
        pop = Population(schema={})
        pop.add_agent(1, extra_col='hello')
        assert pop.size == 1
        assert pop.get_agent_value(1, 'extra_col') == 'hello'

    def test_batch_add_agents(self):
        pop = Population(schema={'wealth': pl.Int64})
        pop.batch_add_agents(3, wealth=0)  # IDs 0, 1, 2
        assert pop.size == 3
        assert pop.get_agent_value(0, 'wealth') == 0
        assert pop.get_agent_value(1, 'wealth') == 0
        assert pop.get_agent_value(2, 'wealth') == 0

    def test_batch_add_agents_with_list_column(self):
        pop = Population(schema={})
        pop.batch_add_agents(3, wealth=[10, 20, 30])
        assert pop.size == 3
        assert pop.get_agent_value(0, 'wealth') == 10
        assert pop.get_agent_value(1, 'wealth') == 20
        assert pop.get_agent_value(2, 'wealth') == 30

    def test_batch_add_agents_length_mismatch(self):
        pop = Population(schema={})
        with pytest.raises(ValueError, match='length mismatch'):
            pop.batch_add_agents(3, wealth=[1, 2])  # length 2 != count 3

    def test_batch_add_agents_scalar_broadcast(self):
        pop = Population(schema={})
        pop.batch_add_agents(5, status='S')
        assert pop.size == 5
        assert pop.get_agent_value(0, 'status') == 'S'
        assert pop.get_agent_value(4, 'status') == 'S'

    def test_batch_update_by_ids(self):
        pop = Population(schema={'wealth': pl.Int64})
        pop.batch_add_agents(3, wealth=0)  # IDs 0, 1, 2

        # Update IDs 0 and 2
        pop.batch_update_by_ids([0, 2], {'wealth': [10, 20]})

        assert pop.get_agent_value(0, 'wealth') == 10
        assert pop.get_agent_value(1, 'wealth') == 0
        assert pop.get_agent_value(2, 'wealth') == 20

    def test_batch_update_by_ids_scalar_value(self):
        pop = Population(schema={'wealth': pl.Int64})
        pop.batch_add_agents(3, wealth=0)
        pop.batch_update_by_ids([0, 1], {'wealth': 99})  # scalar broadcast
        assert pop.get_agent_value(0, 'wealth') == 99
        assert pop.get_agent_value(1, 'wealth') == 99
        assert pop.get_agent_value(2, 'wealth') == 0

    def test_batch_context(self):
        pop = Population(schema={'wealth': pl.Int64})
        pop.batch_add_agents(2, wealth=0)

        with pop.create_batch_context() as batch:
            batch.add_update(0, 'wealth', 50)
            batch.add_update(1, 'wealth', 100)

        assert pop.get_agent_value(0, 'wealth') == 50
        assert pop.get_agent_value(1, 'wealth') == 100

    def test_batch_context_empty(self):
        pop = Population(schema={'wealth': pl.Int64})
        pop.batch_add_agents(2, wealth=0)
        with pop.create_batch_context() as batch:
            pass  # no updates
        # Should not raise
        assert pop.get_agent_value(0, 'wealth') == 0
        assert pop.get_agent_value(1, 'wealth') == 0

    def test_set_agent_value(self):
        pop = Population(schema={'wealth': pl.Int64})
        pop.add_agent(1, wealth=100)
        pop.set_agent_value(1, 'wealth', 200)
        assert pop.get_agent_value(1, 'wealth') == 200

    def test_set_agent_value_new_column(self):
        pop = Population(schema={})
        pop.add_agent(1, wealth=100)
        pop.set_agent_value(1, 'status', 'S')
        assert pop.get_agent_value(1, 'status') == 'S'

    def test_get_agent_value_missing_column_raises(self):
        pop = Population(schema={})
        with pytest.raises(ColumnNotFoundError):
            pop.get_agent_value(999, 'wealth')

    def test_get_agent_value_missing_agent_raises(self):
        pop = Population(schema={'wealth': pl.Int64})
        pop.add_agent(1, wealth=100)
        with pytest.raises(KeyError):
            pop.get_agent_value(999, 'wealth')

    def test_size_empty(self):
        pop = Population(schema={'wealth': pl.Int64})
        assert pop.size == 0

    def test_batch_add_multiple_calls_preserves_ids(self):
        pop = Population(schema={})
        pop.batch_add_agents(2)
        pop.batch_add_agents(3)
        assert pop.size == 5
        ids = pop.data['id'].to_list()
        assert ids == [0, 1, 2, 3, 4]

    @patch('warnings.warn')
    def test_align_and_concat_warns_on_type_fallback(self, mock_warn):
        """When types are irreconcilable, the string fallback should warn."""
        pop = Population(schema={})
        pop.add_agent(1, col=42)  # Int64
        # Adding a value that can't be cast to Int64 should trigger warning
        pop.add_agent(2, col='string_value')
        # The warning should have been issued
        assert mock_warn.called

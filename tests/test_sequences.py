"""
Tests for ambr.sequences module.
"""

import pytest
from unittest.mock import Mock
import ambr as am
from ambr.sequences import AgentList


class TestAgentList:
    """Test cases for AgentList class."""

    def test_agentlist_initialization_empty(self):
        """Test AgentList initialization with empty list."""
        mock_model = Mock()
        agent_list = AgentList(mock_model, [])

        assert agent_list.model == mock_model
        assert len(agent_list) == 0
        assert agent_list.agents == []

    def test_agentlist_initialization_with_agents(self):
        """Test AgentList initialization with agents."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(3)]

        agent_list = AgentList(mock_model, mock_agents)

        assert agent_list.model == mock_model
        assert len(agent_list) == 3
        assert agent_list.agents == mock_agents

    def test_agentlist_len(self):
        """Test AgentList length."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(5)]

        agent_list = AgentList(mock_model, mock_agents)

        assert len(agent_list) == 5

    def test_agentlist_getitem(self):
        """Test AgentList indexing."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(3)]

        agent_list = AgentList(mock_model, mock_agents)

        # Test positive indexing
        assert agent_list[0] == mock_agents[0]
        assert agent_list[1] == mock_agents[1]
        assert agent_list[2] == mock_agents[2]

        # Test negative indexing
        assert agent_list[-1] == mock_agents[2]
        assert agent_list[-2] == mock_agents[1]

    def test_agentlist_getitem_out_of_bounds(self):
        """Test AgentList indexing with out of bounds."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(2)]

        agent_list = AgentList(mock_model, mock_agents)

        with pytest.raises(IndexError):
            agent_list[5]

        with pytest.raises(IndexError):
            agent_list[-5]

    def test_agentlist_setitem(self):
        """Test AgentList item assignment."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(3)]
        new_agent = Mock(id=99)

        agent_list = AgentList(mock_model, mock_agents)
        agent_list[1] = new_agent

        assert agent_list[1] == new_agent
        assert agent_list[1].id == 99

    def test_agentlist_iter(self):
        """Test AgentList iteration."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(3)]

        agent_list = AgentList(mock_model, mock_agents)

        # Test iteration
        iterated_agents = list(agent_list)
        assert iterated_agents == mock_agents

        # Test that iteration works multiple times
        for i, agent in enumerate(agent_list):
            assert agent == mock_agents[i]

    def test_agentlist_contains(self):
        """Test AgentList membership testing."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(3)]
        other_agent = Mock(id=99)

        agent_list = AgentList(mock_model, mock_agents)

        # Test membership
        assert mock_agents[0] in agent_list
        assert mock_agents[1] in agent_list
        assert mock_agents[2] in agent_list
        assert other_agent not in agent_list

    def test_agentlist_append(self):
        """Test AgentList append method."""
        mock_model = Mock()
        agent_list = AgentList(mock_model, [])

        new_agent = Mock(id=1)
        agent_list.append(new_agent)

        assert len(agent_list) == 1
        assert agent_list[0] == new_agent

    def test_agentlist_extend(self):
        """Test AgentList extend method."""
        mock_model = Mock()
        initial_agents = [Mock(id=i) for i in range(2)]
        new_agents = [Mock(id=i) for i in range(10, 12)]

        agent_list = AgentList(mock_model, initial_agents)
        agent_list.extend(new_agents)

        assert len(agent_list) == 4
        assert agent_list[2] == new_agents[0]
        assert agent_list[3] == new_agents[1]

    def test_agentlist_remove(self):
        """Test AgentList remove method."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(3)]

        agent_list = AgentList(mock_model, mock_agents)

        # Remove middle agent
        agent_to_remove = mock_agents[1]
        agent_list.remove(agent_to_remove)

        assert len(agent_list) == 2
        assert agent_to_remove not in agent_list
        assert mock_agents[0] in agent_list
        assert mock_agents[2] in agent_list

    def test_agentlist_remove_not_found(self):
        """Test AgentList remove with agent not in list."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(2)]
        other_agent = Mock(id=99)

        agent_list = AgentList(mock_model, mock_agents)

        with pytest.raises(ValueError):
            agent_list.remove(other_agent)

    def test_agentlist_clear(self):
        """Test AgentList clear method."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(5)]

        agent_list = AgentList(mock_model, mock_agents)
        agent_list.clear()

        assert len(agent_list) == 0
        assert agent_list.agents == []

    def test_agentlist_copy(self):
        """Test AgentList copy method."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(3)]

        agent_list = AgentList(mock_model, mock_agents)
        copied_list = agent_list.copy()

        # Should be different objects
        assert copied_list is not agent_list
        assert copied_list.agents is not agent_list.agents

        # But should have same content
        assert len(copied_list) == len(agent_list)
        assert copied_list.model == agent_list.model
        assert copied_list.agents == agent_list.agents

    def test_agentlist_index(self):
        """Test AgentList index method."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(3)]

        agent_list = AgentList(mock_model, mock_agents)

        # Test finding index
        assert agent_list.index(mock_agents[0]) == 0
        assert agent_list.index(mock_agents[1]) == 1
        assert agent_list.index(mock_agents[2]) == 2

    def test_agentlist_index_not_found(self):
        """Test AgentList index with agent not in list."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(2)]
        other_agent = Mock(id=99)

        agent_list = AgentList(mock_model, mock_agents)

        with pytest.raises(ValueError):
            agent_list.index(other_agent)

    def test_agentlist_count(self):
        """Test AgentList count method."""
        mock_model = Mock()
        agent1 = Mock(id=1)
        agent2 = Mock(id=2)
        agent3 = Mock(id=3)

        # Create list with duplicate
        agent_list = AgentList(mock_model, [agent1, agent2, agent1, agent3])

        assert agent_list.count(agent1) == 2
        assert agent_list.count(agent2) == 1
        assert agent_list.count(agent3) == 1

        # Test with agent not in list
        other_agent = Mock(id=99)
        assert agent_list.count(other_agent) == 0

    def test_agentlist_pop(self):
        """Test AgentList pop method."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(3)]

        agent_list = AgentList(mock_model, mock_agents)

        # Pop last item (default)
        popped = agent_list.pop()
        assert popped == mock_agents[2]
        assert len(agent_list) == 2

        # Pop specific index
        popped = agent_list.pop(0)
        assert popped == mock_agents[0]
        assert len(agent_list) == 1
        assert agent_list[0] == mock_agents[1]

    def test_agentlist_pop_empty(self):
        """Test AgentList pop with empty list."""
        mock_model = Mock()
        agent_list = AgentList(mock_model, [])

        with pytest.raises(IndexError):
            agent_list.pop()

    def test_agentlist_insert(self):
        """Test AgentList insert method."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(2)]
        new_agent = Mock(id=99)

        agent_list = AgentList(mock_model, mock_agents)
        agent_list.insert(1, new_agent)

        assert len(agent_list) == 3
        assert agent_list[0] == mock_agents[0]
        assert agent_list[1] == new_agent
        assert agent_list[2] == mock_agents[1]

    def test_agentlist_reverse(self):
        """Test AgentList reverse method."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(3)]

        agent_list = AgentList(mock_model, mock_agents)
        agent_list.reverse()

        assert len(agent_list) == 3
        assert agent_list[0] == mock_agents[2]
        assert agent_list[1] == mock_agents[1]
        assert agent_list[2] == mock_agents[0]

    def test_agentlist_sort(self):
        """Test AgentList sort method."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in [3, 1, 2]]

        agent_list = AgentList(mock_model, mock_agents)
        agent_list.sort(key=lambda agent: agent.id)

        assert len(agent_list) == 3
        assert agent_list[0].id == 1
        assert agent_list[1].id == 2
        assert agent_list[2].id == 3

    def test_agentlist_repr(self):
        """Test AgentList string representation."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(2)]

        agent_list = AgentList(mock_model, mock_agents)
        repr_str = repr(agent_list)

        assert 'AgentList' in repr_str
        assert '2' in repr_str  # Should show length

    def test_agentlist_empty_operations(self):
        """Test AgentList operations on empty list."""
        mock_model = Mock()
        agent_list = AgentList(mock_model, [])

        # Test operations that should work on empty list
        assert len(agent_list) == 0
        assert list(agent_list) == []
        agent_list.clear()  # Should not raise error
        assert len(agent_list) == 0

        # Test copy of empty list
        copied = agent_list.copy()
        assert len(copied) == 0

    def test_agentlist_slicing(self):
        """Test AgentList slicing operations."""
        mock_model = Mock()
        mock_agents = [Mock(id=i) for i in range(5)]

        agent_list = AgentList(mock_model, mock_agents)

        # Test basic slicing
        slice_result = agent_list[1:3]
        assert len(slice_result) == 2
        assert slice_result == mock_agents[1:3]

        # Test step slicing
        step_result = agent_list[::2]
        assert len(step_result) == 3
        assert step_result == [mock_agents[0], mock_agents[2], mock_agents[4]]


class TestAgentListRecordSubset:
    """Regression tests: record()/update_data() on a subset must not touch other agents."""

    def _make_model(self, n=4):
        from ambr.agent import Agent

        class A(Agent):
            def setup(self):
                pass

        class M(am.Model):
            def setup(self_m):
                for i in range(n):
                    a = A(self_m, i)
                    a.setup()
                    self_m.add_agent(a)

        m = M({})
        m.setup()
        return m

    def test_subset_record_only_touches_subset(self):
        m = self._make_model()
        subset = m.agents[[0, 1]]
        subset.record("flag", 1)
        assert m.agents_df["flag"].to_list() == [1, 1, None, None]

    def test_subset_update_data_only_touches_subset(self):
        import polars as pl

        m = self._make_model()
        m.agents_df = m.agents_df.with_columns(pl.lit(0).alias("y"))
        subset = m.agents[[0, 1]]
        subset.update_data({"y": 99})
        assert m.agents_df["y"].to_list() == [99, 99, 0, 0]

    def test_full_agentlist_record_touches_all(self):
        m = self._make_model()
        m.agents.record("z", 5)
        assert m.agents_df["z"].to_list() == [5, 5, 5, 5]


class TestAgentListViews:
    """Vectorized view API: attribute protocol, filtering, scatter-add."""

    def _make_model(self, n=5):
        import polars as pl
        from ambr.agent import Agent

        class M(am.Model):
            def setup(self_m):
                self_m.add_agents(
                    n,
                    wealth=pl.Series(
                        "wealth", list(range(10, 10 + n)), strict=False
                    ),
                    status="S",
                )

        m = M({"show_progress": False})
        m.setup()
        return m

    def test_column_attr_read_returns_series_from_df(self):
        """``agents.col`` must come from agents_df, not Python attrs.

        Regression for the main footgun: AgentList.__getattr__ used to read
        Python attributes from Agent objects, which silently desynced from
        batch-written DataFrame columns.
        """
        import polars as pl

        m = self._make_model()
        wealth = m.agents.wealth
        assert isinstance(wealth, pl.Series)
        assert wealth.to_list() == [10, 11, 12, 13, 14]

    def test_column_attr_assign_updates_df(self):
        m = self._make_model()
        m.agents.wealth = 0
        assert m.agents_df["wealth"].to_list() == [0, 0, 0, 0, 0]

    def test_column_attr_arith_assign(self):
        m = self._make_model()
        m.agents.wealth -= 5
        assert m.agents_df["wealth"].to_list() == [5, 6, 7, 8, 9]

    def test_where_with_series_predicate(self):
        m = self._make_model()
        view = m.agents.where(m.agents.wealth > 11)
        assert view.ids.to_list() == [2, 3, 4]
        view.status = "R"
        assert m.agents_df["status"].to_list() == ["S", "S", "R", "R", "R"]

    def test_where_with_polars_expression(self):
        import polars as pl

        m = self._make_model()
        view = m.agents.where(pl.col("wealth") >= 13)
        assert view.ids.to_list() == [3, 4]

    def test_getitem_bool_series_is_mask(self):
        import polars as pl

        m = self._make_model()
        mask = pl.Series([True, False, True, False, True])
        view = m.agents[mask]
        assert view.ids.to_list() == [0, 2, 4]

    def test_at_scatter_view(self):
        m = self._make_model()
        view = m.agents.at[[1, 3]]
        view.status = "R"
        assert m.agents_df["status"].to_list() == ["S", "R", "S", "R", "S"]

    def test_scatter_add_with_duplicates(self):
        """scatter_add must SUM deltas when ids repeat (the key flow pattern)."""
        m = self._make_model()
        m.agents.at[[1, 1, 3]].scatter_add(wealth=1)
        assert m.agents_df["wealth"].to_list() == [10, 13, 12, 14, 14]

    def test_iter_yields_underlying_agents(self):
        from ambr.agent import Agent

        class A(Agent):
            def setup(self):
                pass

        class M(am.Model):
            def setup(self_m):
                for i in range(3):
                    a = A(self_m, i)
                    a.setup()
                    self_m.add_agent(a)

        m = M({"show_progress": False})
        m.setup()
        agents = list(m.agents)
        assert len(agents) == 3
        assert all(isinstance(a, A) for a in agents)

    def test_wholesale_assign_from_numpy(self):
        import numpy as np

        m = self._make_model()
        m.agents.wealth = np.array([1, 2, 3, 4, 5])
        assert m.agents_df["wealth"].to_list() == [1, 2, 3, 4, 5]

    def test_view_assignment_after_pending_writes_flush(self):
        """Ensures per-agent write buffer is flushed before bulk writes."""
        m = self._make_model()
        # Queue a per-agent write directly through the buffer — no Agent
        # instances are needed to exercise the flush ordering.
        m._queue_write("wealth", 0, 999)
        assert m._pending_writes.get("wealth", {}).get(0) == 999
        m.agents.wealth = 0
        # Bulk assign should flush then overwrite — result is all 0
        assert m.agents_df["wealth"].to_list() == [0, 0, 0, 0, 0]
        assert m._pending_writes == {}


class TestModelAddAgents:
    """Model.add_agents() bulk-create API."""

    def test_scalar_and_array_columns(self):
        import numpy as np

        class M(am.Model):
            def setup(self_m):
                self_m.add_agents(
                    4,
                    wealth=np.array([10, 20, 30, 40]),
                    status="S",
                )

        m = M({"show_progress": False})
        m.setup()
        assert m.agents_df["wealth"].to_list() == [10, 20, 30, 40]
        assert m.agents_df["status"].to_list() == ["S", "S", "S", "S"]
        assert m.agents_df["id"].to_list() == [0, 1, 2, 3]

    def test_length_mismatch_raises(self):
        class M(am.Model):
            def setup(self_m):
                self_m.add_agents(4, wealth=[1, 2, 3])  # wrong length

        m = M({"show_progress": False})
        with pytest.raises(ValueError, match="length mismatch"):
            m.setup()

    def test_len_reports_df_rows_without_agent_class(self):
        """``len(self.agents)`` must equal the population size even when
        ``add_agents(n)`` was called without an ``agent_class`` (so no
        Python Agent instances were created). Regression guard for a bug
        found while rewriting the benchmark to use the view API."""

        class M(am.Model):
            def setup(self_m):
                self_m.add_agents(7)

        m = M({"show_progress": False})
        m.setup()
        assert len(m.agents) == 7

    def test_agent_class_creates_python_instances(self):
        from ambr.agent import Agent

        class A(Agent):
            def setup(self):
                pass

        class M(am.Model):
            def setup(self_m):
                self_m.add_agents(3, agent_class=A, wealth=[5, 5, 5])

        m = M({"show_progress": False})
        m.setup()
        assert len(m.agents) == 3
        for a in m.agents:
            assert isinstance(a, A)


class TestAgentListIntegration:
    """Integration tests for AgentList with other components."""

    def test_agentlist_with_real_agents(self):
        """Test AgentList with real Agent objects."""
        mock_model = Mock()
        mock_model.p = {'test_param': 'value'}

        # Create real agents
        agents = [am.Agent(mock_model, i) for i in range(3)]
        agent_list = AgentList(mock_model, agents)

        assert len(agent_list) == 3

        # Test that agents are properly accessible
        for i, agent in enumerate(agent_list):
            assert isinstance(agent, am.Agent)
            assert agent.id == i
            assert agent.model == mock_model

    def test_agentlist_with_model_integration(self, basic_model):
        """Test AgentList integration with model."""
        # Create agents for the model
        agents = [am.Agent(basic_model, i) for i in range(5)]
        agent_list = AgentList(basic_model, agents)

        # Test that agents can access model parameters
        for agent in agent_list:
            assert agent.model.p['steps'] == 10  # From fixture

        # Test operations
        assert len(agent_list) == 5
        agent_list.append(am.Agent(basic_model, 99))
        assert len(agent_list) == 6
        assert agent_list[-1].id == 99

    def test_agentlist_performance(self):
        """Test AgentList performance with many agents."""
        mock_model = Mock()

        # Create many agents
        agents = [Mock(id=i) for i in range(1000)]
        agent_list = AgentList(mock_model, agents)

        # Test that operations are still efficient
        assert len(agent_list) == 1000

        # Test iteration performance
        count = 0
        for agent in agent_list:
            count += 1
        assert count == 1000

        # Test access performance
        middle_agent = agent_list[500]
        assert middle_agent.id == 500

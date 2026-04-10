"""
Tests for ambr.agent module.
"""

import pytest
from unittest.mock import Mock, MagicMock
import ambr as am
from ambr.agent import Agent


class TestAgent:
    """Test cases for Agent class."""
    
    def test_agent_initialization(self):
        """Test Agent initialization with model and id."""
        mock_model = Mock()
        mock_model.p = {'test_param': 'value'}
        
        agent = Agent(mock_model, 42)
        
        assert agent.model == mock_model
        assert agent.id == 42
    
    def test_agent_id_types(self):
        """Test Agent with different ID types."""
        mock_model = Mock()
        mock_model.p = {}
        
        # Integer ID
        agent_int = Agent(mock_model, 123)
        assert agent_int.id == 123
        
        # String ID
        agent_str = Agent(mock_model, "agent_001")
        assert agent_str.id == "agent_001"
        
        # UUID-like ID
        agent_uuid = Agent(mock_model, "550e8400-e29b-41d4-a716-446655440000")
        assert agent_uuid.id == "550e8400-e29b-41d4-a716-446655440000"
    
    def test_agent_model_access(self):
        """Test that agent can access model parameters."""
        mock_model = Mock()
        mock_model.p = {
            'simulation_param': 'sim_value',
            'agent_behavior': 'cooperative'
        }
        
        agent = Agent(mock_model, 1)
        
        # Agent should have access to model
        assert agent.model.p['simulation_param'] == 'sim_value'
        assert agent.model.p['agent_behavior'] == 'cooperative'
    
    def test_agent_setup_method(self):
        """Test the setup method (default implementation should do nothing)."""
        mock_model = Mock()
        agent = Agent(mock_model, 1)
        
        # Default setup should not raise any errors
        result = agent.setup()
        assert result is None  # Default implementation returns None
    
    def test_agent_method_delegation(self):
        """Test that agent can call model methods."""
        mock_model = Mock()
        mock_model.some_method = Mock(return_value="model_result")
        
        agent = Agent(mock_model, 1)
        
        # Agent should be able to call model methods
        result = agent.model.some_method("test_arg")
        assert result == "model_result"
        mock_model.some_method.assert_called_once_with("test_arg")
    
    def test_agent_equality(self):
        """Test agent equality based on ID and model."""
        mock_model1 = Mock()
        mock_model2 = Mock()
        
        agent1a = Agent(mock_model1, 1)
        agent1b = Agent(mock_model1, 1)
        agent2 = Agent(mock_model1, 2)
        agent3 = Agent(mock_model2, 1)
        
        # Agents with same model and ID should be considered equal
        assert agent1a.id == agent1b.id
        assert agent1a.model == agent1b.model
        
        # Different IDs should be different
        assert agent1a.id != agent2.id
        
        # Different models should be different  
        assert agent1a.model != agent3.model
    
    def test_agent_repr(self):
        """Test string representation of agent."""
        mock_model = Mock()
        mock_model.__class__.__name__ = "TestModel"
        
        agent = Agent(mock_model, 42)
        repr_str = repr(agent)
        
        assert "Agent" in repr_str
        assert "42" in repr_str
    
    def test_agent_with_none_model(self):
        """Test agent behavior with None model (edge case)."""
        agent = Agent(None, 1)
        assert agent.model is None
        assert agent.id == 1
    
    def test_agent_attribute_storage(self):
        """Test that agents can store custom attributes."""
        mock_model = Mock()
        agent = Agent(mock_model, 1)
        
        # Set custom attributes
        agent.position = (10, 20)
        agent.energy = 100
        agent.state = "active"
        
        # Verify attributes are stored
        assert agent.position == (10, 20)
        assert agent.energy == 100
        assert agent.state == "active"
    
    def test_agent_inheritance(self):
        """Test that Agent can be subclassed properly."""
        mock_model = Mock()
        
        class CustomAgent(Agent):
            def __init__(self, model, agent_id):
                super().__init__(model, agent_id)
                self.custom_attr = "custom_value"
            
            def setup(self):
                self.initialized = True
            
            def custom_method(self):
                return f"Agent {self.id} custom method"
        
        agent = CustomAgent(mock_model, 5)
        
        # Test inheritance
        assert isinstance(agent, Agent)
        assert isinstance(agent, CustomAgent)
        
        # Test custom attributes
        assert agent.custom_attr == "custom_value"
        
        # Test overridden setup
        agent.setup()
        assert agent.initialized is True
        
        # Test custom methods
        assert agent.custom_method() == "Agent 5 custom method"


class TestAgentRecord:
    """Regression tests for Agent.record() / update_data() column handling."""

    def _make_model(self, n=4):
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

    def test_record_creates_new_column(self):
        """Agent.record() must auto-create missing columns (previously crashed with ColumnNotFoundError)."""
        m = self._make_model()
        m.agents[0].record("score", 42)
        assert "score" in m.agents_df.columns
        assert m.agents_df["score"].to_list() == [42, None, None, None]

    def test_record_updates_existing_column(self):
        """Recording on an existing column must preserve other rows."""
        m = self._make_model()
        m.agents[0].record("score", 1)
        m.agents[1].record("score", 2)
        m.agents[2].record("score", 3)
        assert m.agents_df["score"].to_list() == [1, 2, 3, None]

    def test_update_data_creates_new_column(self):
        m = self._make_model()
        m.agents[1].update_data({"age": 7})
        assert m.agents_df["age"].to_list() == [None, 7, None, None]

    def test_update_data_multiple_columns(self):
        m = self._make_model()
        m.agents[0].update_data({"a": 1, "b": 2})
        assert m.agents_df["a"].to_list() == [1, None, None, None]
        assert m.agents_df["b"].to_list() == [2, None, None, None]

    def test_record_is_buffered_until_flush(self):
        """record() should queue, not clone the DataFrame per call."""
        m = self._make_model()
        m.agents[0].record("score", 1)
        # Buffer contains the pending write, df isn't touched yet
        assert m._pending_writes == {"score": {0: 1}}
        assert "score" not in m.population.data.columns
        # Reading via agents_df triggers the flush
        df = m.agents_df
        assert "score" in df.columns
        assert m._pending_writes == {}

    def test_record_last_write_wins_within_step(self):
        m = self._make_model()
        m.agents[0].record("x", 1)
        m.agents[0].record("x", 2)
        m.agents[0].record("x", 3)
        assert m.agents_df["x"].to_list()[0] == 3

    def test_agents_df_setter_clears_buffer(self):
        import polars as pl

        m = self._make_model()
        m.agents[0].record("x", 1)
        m.agents_df = pl.DataFrame({"id": [0, 1, 2, 3], "step": [0, 0, 0, 0]})
        assert m._pending_writes == {}
        assert "x" not in m.agents_df.columns


class TestAgentIntegration:
    """Integration tests for Agent with other components."""

    def test_agent_with_real_model(self, basic_model):
        """Test agent integration with a real model."""
        # Create agent with real model
        agent = Agent(basic_model, 1)
        
        assert agent.model == basic_model
        assert agent.id == 1
        assert agent.model.p['steps'] == 10  # From fixture
    
    def test_multiple_agents_with_same_model(self, basic_model):
        """Test multiple agents sharing the same model."""
        agents = []
        for i in range(5):
            agent = Agent(basic_model, i)
            agents.append(agent)
        
        # All agents should reference the same model
        for agent in agents:
            assert agent.model == basic_model
        
        # But have different IDs
        ids = [agent.id for agent in agents]
        assert ids == [0, 1, 2, 3, 4]
        assert len(set(ids)) == 5  # All unique 
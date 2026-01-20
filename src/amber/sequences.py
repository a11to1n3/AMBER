from typing import List, Type, Callable, Any, Dict, Union
import polars as pl
from .agent import Agent
from .model import Model


class AgentList:
    """Container for managing collections of agents using Polars DataFrames."""

    def __init__(
        self,
        model: Model,
        agents_or_n: Union[List[Agent], int],
        agent_type: Type[Agent] = None,
    ):
        """Initialize a new agent list.

        Args:
            model: The model instance
            agents_or_n: Either a list of existing agents or number of agents to create
            agent_type: Class of agents to create (required if agents_or_n is int)
        """
        self.model = model

        if isinstance(agents_or_n, list):
            # Initialize with existing agents
            self.agents = agents_or_n.copy()
            self.agent_type = agent_type or (
                type(agents_or_n[0]) if agents_or_n else Agent
            )
        else:
            # Create new agents
            if agent_type is None:
                raise ValueError("agent_type is required when creating new agents")
            self.agent_type = agent_type
            self.agents = []
            # Create agents and add them to model's DataFrame
            for i in range(agents_or_n):
                agent = agent_type(model, i)
                agent.setup()
                self.agents.append(agent)

    def __iter__(self):
        """Iterate over agents."""
        return iter(self.agents)

    def __len__(self):
        """Get number of agents."""
        return len(self.agents)

    def __getitem__(self, idx):
        """Get agent by index."""
        return self.agents[idx]

    def __setitem__(self, idx, agent):
        """Set agent by index."""
        self.agents[idx] = agent

    def __contains__(self, agent):
        """Check if agent is in the list."""
        return agent in self.agents

    def __repr__(self):
        """String representation."""
        return f"AgentList({len(self.agents)} agents)"

    def append(self, agent: Agent):
        """Add an agent to the end of the list."""
        self.agents.append(agent)

    def extend(self, agents: List[Agent]):
        """Add multiple agents to the end of the list."""
        self.agents.extend(agents)

    def remove(self, agent: Agent):
        """Remove an agent from the list."""
        self.agents.remove(agent)

    def clear(self):
        """Remove all agents from the list."""
        self.agents.clear()

    def copy(self):
        """Create a shallow copy of the agent list."""
        new_list = AgentList(self.model, [])
        new_list.agents = self.agents.copy()
        new_list.agent_type = self.agent_type
        return new_list

    def index(self, agent: Agent) -> int:
        """Find the index of an agent."""
        return self.agents.index(agent)

    def count(self, agent: Agent) -> int:
        """Count occurrences of an agent."""
        return self.agents.count(agent)

    def pop(self, idx: int = -1) -> Agent:
        """Remove and return an agent at the given index."""
        return self.agents.pop(idx)

    def insert(self, idx: int, agent: Agent):
        """Insert an agent at the given index."""
        self.agents.insert(idx, agent)

    def reverse(self):
        """Reverse the order of agents."""
        self.agents.reverse()

    def sort(self, key=None, reverse=False):
        """Sort the agents."""
        self.agents.sort(key=key, reverse=reverse)

    # Legacy properties for backward compatibility
    @property
    def agent_ids(self):
        """Get list of agent IDs."""
        return [getattr(agent, "id", i) for i, agent in enumerate(self.agents)]

    def select(self, condition: pl.Expr) -> "AgentList":
        """Select agents based on a Polars expression.

        Args:
            condition: Polars expression to filter agents

        Returns:
            New AgentList containing only the selected agents
        """
        selected = AgentList(self.model, [])
        selected.agent_type = self.agent_type
        return selected

    def record(self, name: str, value: Any):
        """Record a variable for all agents in the list.

        Args:
            name: Name of the variable
            value: Value to record
        """
        if hasattr(self.model, "agents_df"):
            self.model.agents_df = self.model.agents_df.with_columns(
                [pl.lit(value).alias(name)]
            )

    def call(self, method_name: str, *args, **kwargs):
        """Call a method on all agents in the list.

        Args:
            method_name: Name of the method to call
            *args: Positional arguments to pass to the method
            **kwargs: Keyword arguments to pass to the method
        """
        for agent in self.agents:
            if hasattr(agent, method_name):
                method = getattr(agent, method_name)
                method(*args, **kwargs)

    def get_data(self) -> pl.DataFrame:
        """Get all agents' data.

        Returns:
            DataFrame containing all agents' data
        """
        if hasattr(self.model, "agents_df"):
            return self.model.agents_df
        return pl.DataFrame()

    def update_data(self, data: Dict[str, Any]):
        """Update data for all agents.

        Args:
            data: Dictionary of column names and values to update
        """
        if hasattr(self.model, "agents_df"):
            for name, value in data.items():
                self.model.agents_df = self.model.agents_df.with_columns(
                    [pl.lit(value).alias(name)]
                )

    def group_by(self, by: str) -> Dict[Any, "AgentList"]:
        """Group agents by a specific column.

        Args:
            by: Name of the column to group by

        Returns:
            Dictionary mapping group values to AgentLists
        """
        groups = {}
        if hasattr(self.model, "agents_df"):
            for group_value, group_df in self.model.agents_df.group_by(by):
                group_list = AgentList(self.model, [])
                group_list.agent_type = self.agent_type
                groups[group_value] = group_list
        return groups

    def apply(self, func: Callable[[Agent], Any]) -> pl.Series:
        """Apply a function to all agents.

        Args:
            func: Function to apply to each agent

        Returns:
            Series containing function results
        """
        results = []
        for agent in self.agents:
            results.append(func(agent))
        return pl.Series(results)

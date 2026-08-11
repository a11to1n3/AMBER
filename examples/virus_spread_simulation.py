#!/usr/bin/env python
# coding: utf-8
"""Interactive AMBER virus-spread (SIR) example.

Core model classes depend only on ``ambr`` / NumPy / Polars and can run
headlessly::

    python examples/virus_spread_simulation.py --headless
    # or: AMBER_VIRUS_HEADLESS=1 python examples/virus_spread_simulation.py

Interactive Plotly + ipywidgets UI requires ``pip install 'ambr[examples]'``
(which includes ``anywidget`` for Plotly ``FigureWidget``) and is constructed
only under ``if __name__ == "__main__":`` (or when explicitly instantiated).

Randomness uses ``model.rng`` only — never the module-global ``random`` module.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from enum import Enum
from typing import Callable, Optional

import ambr as am
import numpy as np
import polars as pl


class HealthStatus(Enum):
    """Health status enumeration for SIR model."""

    SUSCEPTIBLE = "S"
    INFECTED = "I"
    RECOVERED = "R"


class VirusAgent(am.Agent):
    """Agent that can be infected with a virus and spread it to others."""

    def __init__(self, model, agent_id):
        super().__init__(model, agent_id)
        self.status = HealthStatus.SUSCEPTIBLE
        self.infection_time = 0
        world = float(model.p.get("world_size", 100))
        self.x = float(model.rng.uniform(0.0, world))
        self.y = float(model.rng.uniform(0.0, world))

    def setup(self):
        """Initialize agent."""
        # Patient zero - infect a few agents initially
        if self.id < self.model.p.get("initial_infected", 5):
            self.status = HealthStatus.INFECTED
            self.infection_time = 0

    def move(self):
        """Random movement within world boundaries."""
        if self.status == HealthStatus.INFECTED:
            # Infected agents move less (they're sick)
            movement_speed = self.model.p.get("movement_speed", 2.0) * 0.5
        else:
            movement_speed = self.model.p.get("movement_speed", 2.0)

        world_size = float(self.model.p.get("world_size", 100))
        dx = float(self.model.rng.uniform(-movement_speed, movement_speed))
        dy = float(self.model.rng.uniform(-movement_speed, movement_speed))

        self.x = float(max(0.0, min(world_size, self.x + dx)))
        self.y = float(max(0.0, min(world_size, self.y + dy)))

    def interact(self):
        """Check for infections with nearby agents."""
        if self.status != HealthStatus.INFECTED:
            return

        infection_radius = self.model.p.get("infection_radius", 5.0)
        transmission_rate = self.model.p.get("transmission_rate", 0.1)

        for other_id, other_agent in self.model.agent_objects.items():
            if other_id == self.id or other_agent.status != HealthStatus.SUSCEPTIBLE:
                continue

            distance = np.sqrt(
                (self.x - other_agent.x) ** 2 + (self.y - other_agent.y) ** 2
            )

            if distance <= infection_radius:
                if float(self.model.rng.random()) < transmission_rate:
                    other_agent.status = HealthStatus.INFECTED
                    other_agent.infection_time = 0
                    self.model._agents_to_update.add(other_id)

    def update_health(self):
        """Update health status based on infection time."""
        if self.status == HealthStatus.INFECTED:
            self.infection_time += 1
            recovery_time = self.model.p.get("recovery_time", 14)

            if self.infection_time >= recovery_time:
                self.status = HealthStatus.RECOVERED
                self.model._agents_to_update.add(self.id)


class VirusSpreadModel(am.Model):
    """Virus spread model with SIR dynamics."""

    def __init__(self, parameters=None, update_callback: Optional[Callable] = None):
        super().__init__(parameters)
        self.update_callback = update_callback
        self.running = False
        self.paused = False

        self.susceptible_history = []
        self.infected_history = []
        self.recovered_history = []
        self.step_history = []

        self.agent_positions = []
        self.agent_statuses = []

    def setup(self):
        """Initialize model with agents."""
        self.agent_objects = {}
        for i in range(self.p["n"]):
            agent = VirusAgent(self, i)
            agent.setup()
            self.agent_objects[i] = agent

        self.agents = am.AgentList(self, 0, VirusAgent)
        self.agents.agent_ids = list(range(self.p["n"]))

        self._agents_to_update = set()

        self._record_agent_table()
        self._update_history()

    def _record_agent_table(self):
        """Record current agent states into ``agents_df``."""
        agent_data = [
            {
                "id": int(agent_id),
                "step": int(self.t),
                "status": str(agent.status.value),
                "x": float(agent.x),
                "y": float(agent.y),
                "infection_time": int(agent.infection_time),
            }
            for agent_id, agent in self.agent_objects.items()
        ]

        if agent_data:
            self.agents_df = pl.DataFrame(agent_data)

    def _update_history(self):
        """Update history for real-time plotting and trigger callback."""
        susceptible = sum(
            1
            for agent in self.agent_objects.values()
            if agent.status == HealthStatus.SUSCEPTIBLE
        )
        infected = sum(
            1
            for agent in self.agent_objects.values()
            if agent.status == HealthStatus.INFECTED
        )
        recovered = sum(
            1
            for agent in self.agent_objects.values()
            if agent.status == HealthStatus.RECOVERED
        )

        self.susceptible_history.append(susceptible)
        self.infected_history.append(infected)
        self.recovered_history.append(recovered)
        self.step_history.append(self.t)

        positions = [(agent.x, agent.y) for agent in self.agent_objects.values()]
        statuses = [agent.status.value for agent in self.agent_objects.values()]

        self.agent_positions = positions
        self.agent_statuses = statuses

        if self.update_callback:
            self.update_callback(self)

    def step(self):
        """Execute one simulation step body (called via ``run_step``)."""
        if self.paused:
            return

        self._agents_to_update.clear()

        for agent in self.agent_objects.values():
            agent.move()
            agent.interact()
            agent.update_health()

        if self._agents_to_update:
            self._append_agent_snapshots()

    def _append_agent_snapshots(self):
        """Append current state rows for agents touched this step."""
        agent_data = [
            {
                "id": int(agent_id),
                "step": int(self.t),
                "status": str(self.agent_objects[agent_id].status.value),
                "x": float(self.agent_objects[agent_id].x),
                "y": float(self.agent_objects[agent_id].y),
                "infection_time": int(self.agent_objects[agent_id].infection_time),
            }
            for agent_id in self._agents_to_update
        ]

        if agent_data:
            new_data = pl.DataFrame(agent_data)
            self.agents_df = pl.concat([self.agents_df, new_data])

    def update(self):
        """Post-step hook: history + optional FPS throttle."""
        self._update_history()

        fps = self.p.get("fps", 0)
        if fps and fps > 0:
            time.sleep(1.0 / float(fps))

    def pause(self):
        """Pause the simulation."""
        self.paused = True

    def resume(self):
        """Resume the simulation."""
        self.paused = False

    def reset(self):
        """Reset simulation to initial state."""
        self.t = 0
        self.paused = False
        self.running = False
        self._setup_done = False

        world = float(self.p.get("world_size", 100))
        for i, agent in enumerate(self.agent_objects.values()):
            agent.status = HealthStatus.SUSCEPTIBLE
            agent.infection_time = 0
            agent.x = float(self.rng.uniform(0.0, world))
            agent.y = float(self.rng.uniform(0.0, world))

            if i < self.p.get("initial_infected", 5):
                agent.status = HealthStatus.INFECTED
                agent.infection_time = 0

        self.susceptible_history = []
        self.infected_history = []
        self.recovered_history = []
        self.step_history = []
        self.agent_positions = []
        self.agent_statuses = []

        self.agents_df = pl.DataFrame()
        self._record_agent_table()
        self._update_history()
        self._setup_done = True

    def end(self):
        """Finalize simulation."""
        self.running = False


def run_headless(
    steps: int = 3,
    n: int = 40,
    seed: int = 0,
    **overrides,
) -> VirusSpreadModel:
    """Run a short non-interactive virus simulation (CI / smoke).

    Does not import plotly, ipywidgets, or anywidget.
    """
    params = {
        "n": n,
        "steps": steps,
        "seed": seed,
        "show_progress": False,
        "transmission_rate": 0.15,
        "recovery_time": 5,
        "movement_speed": 2.0,
        "infection_radius": 5.0,
        "initial_infected": 3,
        "fps": 0,
        "world_size": 50,
    }
    params.update(overrides)
    model = VirusSpreadModel(params)
    for _ in range(int(steps)):
        model.run_step()
    model.end()
    return model


class VirusSpreadSimulation:
    """Interactive virus-spread UI (requires ``ambr[examples]``)."""

    def __init__(self):
        self.model = None
        self.simulation_thread = None
        self.running = False
        self._last_error: Optional[BaseException] = None
        self._create_interface()

    def _create_interface(self):
        """Create the complete interactive interface."""
        import ipywidgets as widgets

        self._widgets = widgets

        style = {"description_width": "140px"}
        layout = widgets.Layout(width="300px")

        self.population_slider = widgets.IntSlider(
            value=500,
            min=50,
            max=2000,
            step=50,
            description="Population:",
            style=style,
            layout=layout,
        )

        self.transmission_rate_slider = widgets.FloatSlider(
            value=0.05,
            min=0.01,
            max=0.3,
            step=0.01,
            description="Transmission Rate:",
            style=style,
            layout=layout,
        )

        self.recovery_time_slider = widgets.IntSlider(
            value=14,
            min=5,
            max=30,
            step=1,
            description="Recovery Time:",
            style=style,
            layout=layout,
        )

        self.movement_speed_slider = widgets.FloatSlider(
            value=2.0,
            min=0.5,
            max=5.0,
            step=0.1,
            description="Movement Speed:",
            style=style,
            layout=layout,
        )

        self.infection_radius_slider = widgets.FloatSlider(
            value=5.0,
            min=1.0,
            max=15.0,
            step=0.5,
            description="Infection Radius:",
            style=style,
            layout=layout,
        )

        self.initial_infected_slider = widgets.IntSlider(
            value=5,
            min=1,
            max=20,
            step=1,
            description="Initial Infected:",
            style=style,
            layout=layout,
        )

        self.fps_slider = widgets.IntSlider(
            value=15,
            min=1,
            max=30,
            step=1,
            description="FPS:",
            style=style,
            layout=layout,
        )

        self.max_steps_slider = widgets.IntSlider(
            value=300,
            min=100,
            max=1000,
            step=50,
            description="Max Steps:",
            style=style,
            layout=layout,
        )

        button_layout = widgets.Layout(width="90px", height="35px")

        self.start_button = widgets.Button(
            description="▶️ Start",
            button_style="success",
            layout=button_layout,
            tooltip="Start simulation",
        )

        self.pause_button = widgets.Button(
            description="⏸️ Pause",
            button_style="warning",
            layout=button_layout,
            tooltip="Pause/Resume simulation",
        )

        self.reset_button = widgets.Button(
            description="🔄 Reset",
            button_style="info",
            layout=button_layout,
            tooltip="Reset to initial state",
        )

        self.status_display = widgets.HTML(
            value=(
                "<div style='font-size: 14px; font-weight: bold;'>"
                "Status: <span style='color: #666;'>Ready</span></div>"
            )
        )

        self.step_display = widgets.HTML(
            value=(
                "<div style='font-size: 14px;'>Step: "
                "<span style='color: #007acc;'>0</span></div>"
            )
        )

        self.infected_display = widgets.HTML(
            value=(
                "<div style='font-size: 14px;'>Infected: "
                "<span style='color: #d73027;'>0</span></div>"
            )
        )

        self.recovered_display = widgets.HTML(
            value=(
                "<div style='font-size: 14px;'>Recovered: "
                "<span style='color: #1a9641;'>0</span></div>"
            )
        )

        self._create_plots()

        self.start_button.on_click(self._start_simulation)
        self.pause_button.on_click(self._pause_simulation)
        self.reset_button.on_click(self._reset_simulation)

        self._create_layout()

    def _create_plots(self):
        """Create Plotly FigureWidget visualizations (needs anywidget)."""
        import plotly.graph_objects as go

        self.sir_figure = go.FigureWidget()

        self.sir_figure.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Susceptible",
                line=dict(color="#1f77b4", width=3),
            )
        )

        self.sir_figure.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Infected",
                line=dict(color="#d62728", width=3),
            )
        )

        self.sir_figure.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="lines",
                name="Recovered",
                line=dict(color="#2ca02c", width=3),
            )
        )

        self.sir_figure.update_layout(
            title=dict(text="<b>SIR Epidemic Curves</b>", font=dict(size=16)),
            xaxis_title="Time Step",
            yaxis_title="Number of Agents",
            height=350,
            margin=dict(l=60, r=30, t=60, b=50),
            plot_bgcolor="rgba(240,240,240,0.3)",
            legend=dict(x=0.7, y=0.95),
        )

        self.spatial_figure = go.FigureWidget()
        self.spatial_figure.add_trace(
            go.Scatter(
                x=[],
                y=[],
                mode="markers",
                marker=dict(
                    size=8,
                    color=[],
                    colorscale=[
                        [0, "#1f77b4"],
                        [0.5, "#d62728"],
                        [1, "#2ca02c"],
                    ],
                    cmin=0,
                    cmax=2,
                    showscale=False,
                ),
                name="Agents",
            )
        )

        self.spatial_figure.update_layout(
            title=dict(text="<b>Spatial Distribution</b>", font=dict(size=16)),
            xaxis_title="X Position",
            yaxis_title="Y Position",
            height=350,
            margin=dict(l=60, r=30, t=60, b=50),
            plot_bgcolor="rgba(240,240,240,0.3)",
            showlegend=False,
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100]),
        )

    def _create_layout(self):
        """Create the responsive layout."""
        widgets = self._widgets

        control_panel = widgets.VBox(
            [
                widgets.HTML(
                    value=(
                        "<h3 style='margin-bottom: 20px; color: #333;'>"
                        "🦠 Epidemic Controls</h3>"
                    )
                ),
                self.population_slider,
                self.transmission_rate_slider,
                self.recovery_time_slider,
                self.movement_speed_slider,
                self.infection_radius_slider,
                self.initial_infected_slider,
                self.fps_slider,
                self.max_steps_slider,
                widgets.HTML("<div style='margin: 15px 0;'></div>"),
                widgets.HBox(
                    [self.start_button, self.pause_button, self.reset_button],
                    layout=widgets.Layout(justify_content="space-between"),
                ),
                widgets.HTML(
                    value=(
                        "<h4 style='margin: 20px 0 10px 0; color: #333;'>"
                        "📊 Status</h4>"
                    )
                ),
                self.status_display,
                self.step_display,
                self.infected_display,
                self.recovered_display,
            ],
            layout=widgets.Layout(
                width="340px",
                padding="20px",
                border="1px solid #ddd",
                border_radius="8px",
                background_color="#fafafa",
            ),
        )

        plots_panel = widgets.VBox(
            [self.sir_figure, self.spatial_figure],
            layout=widgets.Layout(padding="20px"),
        )

        self.interface = widgets.HBox(
            [control_panel, plots_panel],
            layout=widgets.Layout(
                border="2px solid #d62728",
                border_radius="10px",
                padding="10px",
                background_color="white",
            ),
        )

    def _get_parameters(self):
        """Get current parameter values from controls."""
        return {
            "n": self.population_slider.value,
            "transmission_rate": self.transmission_rate_slider.value,
            "recovery_time": self.recovery_time_slider.value,
            "movement_speed": self.movement_speed_slider.value,
            "infection_radius": self.infection_radius_slider.value,
            "initial_infected": self.initial_infected_slider.value,
            "fps": self.fps_slider.value,
            "steps": self.max_steps_slider.value,
            "world_size": 100,
            "show_progress": False,
        }

    def _update_visualizations(self, model):
        """Update plots with current model data."""
        with self.sir_figure.batch_update():
            self.sir_figure.data[0].x = model.step_history
            self.sir_figure.data[0].y = model.susceptible_history
            self.sir_figure.data[1].x = model.step_history
            self.sir_figure.data[1].y = model.infected_history
            self.sir_figure.data[2].x = model.step_history
            self.sir_figure.data[2].y = model.recovered_history

        if model.agent_positions:
            x_coords = [pos[0] for pos in model.agent_positions]
            y_coords = [pos[1] for pos in model.agent_positions]

            status_colors = []
            for status in model.agent_statuses:
                if status == "S":
                    status_colors.append(0)
                elif status == "I":
                    status_colors.append(1)
                else:
                    status_colors.append(2)

            with self.spatial_figure.batch_update():
                self.spatial_figure.data[0].x = x_coords
                self.spatial_figure.data[0].y = y_coords
                self.spatial_figure.data[0].marker.color = status_colors

        self.step_display.value = (
            f"<div style='font-size: 14px;'>Step: "
            f"<span style='color: #007acc;'>{model.t}</span></div>"
        )

        if model.infected_history:
            infected_count = model.infected_history[-1]
            recovered_count = model.recovered_history[-1]

            self.infected_display.value = (
                f"<div style='font-size: 14px;'>Infected: "
                f"<span style='color: #d73027;'>{infected_count}</span></div>"
            )
            self.recovered_display.value = (
                f"<div style='font-size: 14px;'>Recovered: "
                f"<span style='color: #1a9641;'>{recovered_count}</span></div>"
            )

    def _model_update_callback(self, model):
        """Callback function for real-time model updates."""
        self._update_visualizations(model)

    def _set_status(self, label: str, color: str) -> None:
        self.status_display.value = (
            f"<div style='font-size: 14px; font-weight: bold;'>"
            f"Status: <span style='color: {color};'>{label}</span></div>"
        )

    def _start_simulation(self, button):
        """Start simulation in background thread."""
        if self.running:
            return

        params = self._get_parameters()
        self.model = VirusSpreadModel(
            parameters=params,
            update_callback=self._model_update_callback,
        )

        self._last_error = None
        self._set_status("Running", "#28a745")
        self.running = True

        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def _run_simulation(self):
        """Main simulation execution loop (uses ``run_step`` lifecycle)."""
        error: Optional[BaseException] = None
        try:
            # First run_step runs setup(); do not call step()/update() by hand.
            while self.model.t < self.model.p["steps"] and self.running:
                if not self.model.paused:
                    self.model.run_step()

                    if (
                        self.model.infected_history
                        and self.model.infected_history[-1] == 0
                        and self.model.t > 0
                    ):
                        break
                else:
                    time.sleep(0.1)

            self.model.end()

        except Exception as exc:
            error = exc
            self._last_error = exc
            print(f"❌ Simulation error: {exc}", file=sys.stderr)
        finally:
            self.running = False
            if error is not None:
                # Surface failures — never claim "Completed" on error.
                msg = f"Failed: {type(error).__name__}: {error}"
                self._set_status(msg, "#dc3545")
            else:
                self._set_status("Completed", "#28a745")

    def _pause_simulation(self, button):
        """Pause or resume the simulation."""
        if not self.model:
            return

        if self.model.paused:
            self.model.resume()
            self.pause_button.description = "⏸️ Pause"
            self._set_status("Running", "#28a745")
        else:
            self.model.pause()
            self.pause_button.description = "▶️ Resume"
            self._set_status("Paused", "#ffc107")

    def _reset_simulation(self, button):
        """Reset simulation to initial conditions."""
        self.running = False
        self._last_error = None

        if self.model:
            self.model.reset()
            self._update_visualizations(self.model)

        self._set_status("Ready", "#666")
        self.pause_button.description = "⏸️ Pause"

    def display(self):
        """Display the interactive interface (notebook-friendly)."""
        return self.interface


def _main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="AMBER virus-spread example")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run a short non-interactive smoke (no plotly/widgets)",
    )
    parser.add_argument("--steps", type=int, default=3, help="Headless step count")
    parser.add_argument("--n", type=int, default=40, help="Population size")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args(argv)

    headless = args.headless or os.environ.get("AMBER_VIRUS_HEADLESS", "").strip() in {
        "1",
        "true",
        "yes",
    }

    if headless:
        model = run_headless(steps=args.steps, n=args.n, seed=args.seed)
        print(
            f"headless ok: t={model.t} "
            f"S={model.susceptible_history[-1]} "
            f"I={model.infected_history[-1]} "
            f"R={model.recovered_history[-1]}"
        )
        return 0

    # Interactive UI — requires ambr[examples]
    try:
        sim = VirusSpreadSimulation()
    except ImportError as exc:
        print(
            "Interactive UI requires: pip install 'ambr[examples]'\n"
            f"Underlying error: {exc}\n"
            "Or run headless: python examples/virus_spread_simulation.py --headless",
            file=sys.stderr,
        )
        return 1

    try:
        from IPython.display import display as ipy_display

        ipy_display(sim.display())
    except Exception:
        # Plain script: interface object exists; user can open in a notebook.
        print(
            "VirusSpreadSimulation UI constructed. "
            "In a Jupyter notebook call VirusSpreadSimulation().display()."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

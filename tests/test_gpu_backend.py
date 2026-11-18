"""Model.gpu() / Model.cpu() on the vectorized view API."""

import numpy as np
import polars as pl
import pytest

import ambr as am
from ambr.execution import ExecutionConfig, begin_execution, end_execution, get_device_column
from ambr.gpu import GPU_AVAILABLE, to_host


@pytest.fixture(params=["emulated", "cuda"])
def gpu_exports(request, monkeypatch):
    """Run the same dispatch tests against a CPU double and real CuPy."""
    from ambr import execution

    if request.param == "cuda" and not GPU_AVAILABLE:
        pytest.skip("CUDA/CuPy not available")
    if request.param == "emulated":
        monkeypatch.setattr(execution, "require_gpu", lambda: None)
        monkeypatch.setattr(execution, "get_array_module", lambda _: np)
        monkeypatch.setattr(execution, "to_device", np.copy)
        monkeypatch.setattr(execution, "to_host", np.copy)
        monkeypatch.setattr(execution, "make_device_rng", np.random.default_rng)
        monkeypatch.setattr(execution, "synchronize", lambda: None)
    exports = []
    export = execution.to_host

    def observed_export(array):
        exports.append(array.shape)
        return export(array)

    monkeypatch.setattr(execution, "to_host", observed_export)
    monkeypatch.setattr("ambr._auto_batch.gettrace", lambda: None)
    return exports


class _Person(am.Agent):
    def setup(self):
        self.age = self.id
        self.active = self.id % 4 == 0
        self.energy = 100

    def step(self):
        if self.active:
            self.age += 1

    def use_energy(self):
        if self.id % 3 == 1:
            self.energy -= 2


def test_repeated_device_view_targets_require_a_reduction(gpu_exports):
    class Model(am.Model):
        def setup(self):
            self.add_agents(2, x=[4, 8])

        def step(self):
            targets = self.agents.at[self.xp.asarray([0, 0])]
            with pytest.raises(ValueError, match="unique target IDs"):
                targets.x = 100
            with pytest.raises(ValueError, match="unique target IDs"):
                targets.set(x=100)
            targets.scatter_add(x=1)

    result = Model({"steps": 1}).gpu().run(contract="check")
    assert result.agents["id"].to_list() == [0, 1]
    assert result.agents["x"].to_list() == [6, 8]
    cert = result["contract"][0]
    assert cert.clean and cert.conditional_guarantee == "not_established"


def test_keyed_draws_match_cpu_with_the_same_gpu_model_declaration(gpu_exports):
    class Person(am.Agent):
        def step(self):
            if self.draw("move", index=2) < 0.45:
                self.x += 1

    class Model(am.Model):
        def setup(self):
            self.add_agents(96, agent_class=Person, x=0)

        def step(self):
            self.agents.step()

    params = {"seed": 11, "steps": 4}
    scalar = Model(params).run(auto_batch=False)
    assert len(scalar.agents["x"].unique()) > 1
    for model, device in (
        (Model(params), "cpu"),
        (Model(params).vectorize(), "cpu"),
        (Model(params).gpu(), "gpu"),
    ):
        result = model.run()
        assert result.info["device"] == device
        assert result.agents.equals(scalar.agents)
        assert result.info["auto_batch"] == {"enabled": True, "batches": 4, "agent_calls": 384}
    assert gpu_exports == [(96,)] * 2
    monitored = Model(params).gpu().run(contract="check")
    assert monitored.agents.equals(scalar.agents)
    assert monitored.info["auto_batch"]["batches"] == 0
    assert all(c.clean and c.conditional_guarantee == "not_established" for c in monitored["contract"])


def test_keyed_batching_respects_custom_draw_methods(gpu_exports):
    class Person(am.Agent):
        def draw(self, event):
            return 0.0

        def step(self):
            if self.draw("move") < 0.45:
                self.x += 1

    class Model(am.Model):
        def setup(self):
            self.add_agents(96, agent_class=Person, x=0)

        def step(self):
            self.agents.step()

    result = Model({"seed": 11, "steps": 4}).gpu().run()
    assert result.agents["x"].to_list() == [4] * 96
    assert result.info["auto_batch"]["batches"] == 0


def test_bounded_keyed_walk_batches_and_matches_every_step(gpu_exports):
    class Walker(am.Agent):
        def step(self):
            self.x = max(0.0, min(8.0, self.x + (self.draw("move") * 2.0 - 1.0)))
            self.y = max(0.0, min(5.0, self.y + (self.draw("move", index=1) * 4.0 - 2.0)))

    class Model(am.Model):
        agent_reporters = ["x", "y"]

        def setup(self):
            self.add_agents(96, agent_class=Walker, x=0.0, y=5.0)

        def step(self):
            self.agents.step()

    params = {"seed": 42, "steps": 5}
    scalar = Model(params).run(auto_batch=False)
    assert scalar.agent_vars["x"].min() == 0.0 < scalar.agent_vars["x"].max()
    assert scalar.agent_vars["y"].min() < scalar.agent_vars["y"].max() == 5.0
    for model in (Model(params), Model(params).vectorize(), Model(params).gpu()):
        result = model.run()
        assert result.agent_vars.equals(scalar.agent_vars)
        assert result.agents.equals(scalar.agents)
        assert result.info["auto_batch"] == {"enabled": True, "batches": 5, "agent_calls": 480}


def test_bounded_keyed_walk_respects_custom_min(gpu_exports, monkeypatch):
    class Walker(am.Agent):
        def step(self):
            self.x = max(0.0, min(8.0, self.x + (self.draw("move") * 2.0 - 1.0)))

    class Model(am.Model):
        def setup(self):
            self.add_agents(96, agent_class=Walker, x=0.0)

        def step(self):
            self.agents.step()

    monkeypatch.setitem(Walker.step.__globals__, "min", lambda *_: 3.0)
    result = Model({"seed": 42, "steps": 1}).gpu().run()
    assert result.info["auto_batch"]["batches"] == 0
    assert result.agents["x"].to_list() == [3.0] * 96


class _AgentModel(am.Model):
    def setup(self):
        self.agents = am.AgentList(self, 96, _Person)

    def step_oop(self):
        if self.p.dispatch == "activate":
            self.activate_agents()
        elif self.p.dispatch == "call":
            assert self.agents.call("step").tolist() == [None] * 96
        else:
            assert self.agents.step().tolist() == [None] * 96
        self.agents.use_energy()


def test_same_model_declaration_runs_with_each_execution_selector(gpu_exports):
    class Model(am.Model):
        def setup(self):
            self.add_agents(96, agent_class=_Person)

        def step(self):
            self.agents.step()

    params = {"steps": 3}
    reference = Model(params).run(auto_batch=False)
    for configure, device in (
        (lambda model: model, "cpu"),
        (lambda model: model.vectorize(), "cpu"),
        (lambda model: model.gpu(), "gpu"),
        (lambda model: model.vectorize().gpu(), "gpu"),
        (lambda model: model.gpu().vectorize(), "gpu"),
    ):
        model = configure(Model(params))
        result = model.run()
        assert result.info["device"] == device
        assert result.info["auto_batch"]["batches"] == 3
        assert result.agents.equals(reference.agents)
        assert model.agents.numpy("step").tolist() == [0] * 96
        assert model.agents.array("step").tolist() == [0] * 96


@pytest.mark.parametrize("dispatch", ["activate", "call", "broadcast"])
def test_gpu_batches_ordinary_agent_methods_without_per_step_exports(dispatch, gpu_exports):
    params = {"steps": 3, "dispatch": dispatch}
    cpu = _AgentModel(params).run(auto_batch=False)
    gpu_model = _AgentModel(params)
    gpu = gpu_model.gpu().run()
    assert gpu.info["device"] == "gpu"
    assert gpu.info["auto_batch"] == {"enabled": True, "batches": 6, "agent_calls": 576}
    assert gpu.agents.equals(cpu.agents)
    assert vars(gpu_model.agents.by_id(0))["age"] == 3
    # Each changed column and its assignment mask is downloaded once.
    assert gpu_exports == [(96,)] * 4


def test_gpu_batch_and_scalar_rules_share_state(gpu_exports):
    class Person(_Person):
        def follow_previous(self):
            if self.id:
                self.age = self.model.agents.by_id(self.id - 1).age + 1

    class Model(am.Model):
        agent_reporters = ["age", "energy"]
        record_initial = True

        def setup(self):
            self.add_agents(96, agent_class=Person)

        def step(self):
            self.activate_agents()
            self.agents.by_id(0).age += 10
            self.agents.call("follow_previous")
            self.agents.use_energy()
            assert vars(self.agents.by_id(0))["age"] == (self.t + 1) * 11

        def end(self):
            self.end_age = self.agents.by_id(95).age
            self.end_energy = self.agents_df["energy"].sum()

    cpu_model = Model({"steps": 3})
    cpu = cpu_model.run(auto_batch=False)
    gpu_model = Model({"steps": 3})
    gpu = gpu_model.gpu().run()
    assert gpu.info["auto_batch"]["batches"] == 6
    assert gpu.agents.equals(cpu.agents)
    assert gpu.agent_vars.equals(cpu.agent_vars)
    assert (gpu_model.end_age, gpu_model.end_energy) == (128, 9408)


@pytest.mark.parametrize("add_column", [False, True])
def test_empty_host_scatter_preserves_prior_gpu_batch(add_column, gpu_exports):
    class Counter(am.Agent):
        def step(self):
            self.x += 1

    class Model(am.Model):
        def setup(self):
            self.add_agents(64, agent_class=Counter, x=3)

        def step(self):
            self.activate_agents()
            empty = self.agents.where(pl.Series([False] * 64))
            if add_column:
                empty.scatter_add(x=1, extra=1)
            else:
                empty.scatter_add(x=1)

    args = {"steps": 1}
    cpu = Model(args).run(auto_batch=False)
    gpu = Model(args).gpu().run()
    assert gpu.info["auto_batch"]["batches"] == 1
    assert gpu.agents.equals(cpu.agents)


def test_gpu_batch_observes_in_place_frame_edits(gpu_exports):
    class Counter(am.Agent):
        def step(self):
            self.x += 1
            self.y += 1

        def advance(self):
            self.y += 1

    class Model(am.Model):
        def setup(self):
            self.add_agents(64, agent_class=Counter, x=0, y=0)

        def step(self):
            frame = self.agents_df
            self.agents.advance()
            frame.replace_column(frame.columns.index("x"), pl.Series("x", [100] * 64))
            assert self.agents.by_id(0).x == 100
            self.agents.step()

    args = {"steps": 1}
    cpu = Model(args).run(auto_batch=False)
    gpu = Model(args).gpu().run()
    assert gpu.info["auto_batch"]["batches"] == 2
    assert gpu.agents.equals(cpu.agents)
    assert gpu.agents["x"].to_list() == [101] * 64
    assert gpu.agents["y"].to_list() == [2] * 64


@pytest.mark.parametrize("activation", ["sequential", "random"])
def test_gpu_replay_preserves_the_forward_device_stream(activation, gpu_exports):
    class Writer(am.Agent):
        def step(self):
            self.model.agents.by_id(0).x = self.id

    class Model(am.Model):
        def setup(self):
            self.add_agents(2, agent_class=Writer, x=0)

        def step(self):
            self.activate_agents(self.p.activation)
            self.record_model("draw", self.rng.integers(2**31).item())
            self.record_model("binomial", int(self.rng.binomial(10, 0.5)))

        def end(self):
            self.end_draw = self.rng.integers(2**31).item()
            self.end_binomial = int(self.rng.binomial(10, 0.5))

    params = {"steps": 1, "seed": 43, "activation": activation}
    plain_model = Model(params).gpu()
    checked_model = Model(params).gpu()
    plain = plain_model.run(contract="off")
    checked = checked_model.run(contract="check")
    races = [v for v in checked["contract"][0].violations if v.kind == "activation_race"]
    assert [v.divergence_witness for v in races] == [True]
    assert checked.agents.equals(plain.agents)
    assert checked["model"].equals(plain["model"])
    assert checked_model.end_draw == plain_model.end_draw
    assert checked_model.end_binomial == plain_model.end_binomial


def test_gpu_batch_preserves_mask_after_later_column_assignments(gpu_exports):
    class Model(_AgentModel):
        def step_oop(self):
            self.agents.step()
            self.agents.active = False
            self.agents.age = 50
            assert self.agents.by_id(0).age == 50
            assert vars(self.agents.by_id(0))["age"] == 1
            assert vars(self.agents.by_id(1))["age"] == 1

    result = Model({"steps": 1}).gpu().run()
    assert result.info["auto_batch"]["batches"] == 1
    assert result.agents["age"].to_list() == [50] * 96


def test_gpu_batch_composes_with_buffered_scalar_writes(gpu_exports):
    class Model(_AgentModel):
        def step_oop(self):
            self.activate_agents()
            self.agents.by_id(0).age = 80
            self.activate_agents()

    cpu = Model({"steps": 1}).run(auto_batch=False)
    gpu = Model({"steps": 1}).gpu().run()
    assert gpu.info["auto_batch"]["batches"] == 2
    assert gpu.agents.equals(cpu.agents)
    assert gpu.agents["age"][0] == 81
    assert gpu.agents["age"][4] == 6


@pytest.mark.parametrize("bulk", [False, True])
def test_gpu_batch_survives_population_growth(bulk, gpu_exports):
    class Model(am.Model):
        def setup(self):
            self.add_agents(96, agent_class=_Person)

        def step(self):
            self.activate_agents()
            if self.t == 0:
                if bulk:
                    self.add_agents(1, agent_class=_Person)
                else:
                    person = _Person(self, 96)
                    person.setup()
                    self.add_agent(person)
            self.activate_agents()

    cpu = Model({"steps": 2}).run(auto_batch=False)
    gpu = Model({"steps": 2}).gpu().run()
    assert gpu.info["auto_batch"]["batches"] == 4
    assert gpu.agents.equals(cpu.agents)
    assert gpu.agents["age"][96] == 99


def test_gpu_random_activation_uses_the_python_schedule(gpu_exports):
    class Person(_Person):
        def remember_visit(self):
            self.model.order.append(self.id)

    class Model(_AgentModel):
        def setup(self):
            self.agents = am.AgentList(self, 96, Person)
            self.order = []

        def step_oop(self):
            self.activate_agents(mode="random", method="remember_visit")

    params = {"steps": 2, "seed": 31}
    cpu = Model(params)
    gpu = Model(params)
    cpu.run()
    result = gpu.gpu().run()
    assert result.info["auto_batch"]["batches"] == 0
    assert gpu.order == cpu.order
    assert len(gpu.order) == 192


def test_gpu_batch_overflow_falls_back_without_partial_updates(gpu_exports):
    class Person(_Person):
        def step(self):
            self.age += 1
            self.energy -= 1

    class Model(am.Model):
        def setup(self):
            self.add_agents(96, agent_class=Person)
            self.agents.by_id(0).energy = -(1 << 63)

        def step(self):
            self.activate_agents()
            assert self.agents.by_id(0).energy == -(1 << 63) - 1
            # Restore a representable value before committing the frame.
            self.agents.by_id(0).energy += 1

    result = Model({"steps": 1}).gpu().run()
    assert result.info["auto_batch"]["batches"] == 0
    assert result.agents["age"].to_list() == list(range(1, 97))
    assert result.agents["energy"].to_list() == [-(1 << 63)] + [99] * 95


def test_gpu_end_observes_current_host_frame_and_can_reduce(gpu_exports):
    class Model(am.Model):
        def setup(self):
            self.add_agents(3, x=0)

        def step(self):
            self.agents.x = 7

        def end(self):
            self.seen_in_end = self.agents_df["x"].to_list()
            self.agents.at[[0, 0]].scatter_max(x=[6, 9])
            self.agents.at[[1, 1]].scatter_max(x=[10, 11], new=[1, 2])

    model = Model({"steps": 1})
    result = model.gpu().run()
    assert model.seen_in_end == [7, 7, 7]
    assert result.agents["x"].to_list() == [9, 11, 7]
    assert result.agents["new"].to_list() == [0, 2, 0]


def test_gpu_frame_reports_and_view_ids_and_retained_end_writes(gpu_exports):
    class Model(am.Model):
        def setup(self):
            self.add_agents(3, x=0, y=0)

        def step(self):
            self.held = self.agents.x.array
            self.held[:] = 7
            view = self.agents.at[self.xp.asarray([1, 2])]
            assert view.ids.to_list() == [1, 2]
            view.y = [9, 8]
            self.record_model("sum", self.agents_df["x"].sum())

        def end(self):
            self.held[0] = 9

    result = Model({"steps": 1}).gpu().run()
    assert result.model["sum"].to_list() == [21]
    assert result.agents.select("id", "x", "y").rows() == [(0, 9, 0), (1, 7, 9), (2, 7, 8)]


def test_gpu_batch_preserves_unrelated_pending_values_and_subview_reads(gpu_exports):
    class Model(_AgentModel):
        def step_oop(self):
            self.agents[0].age = 10
            self.agents[0].energy = np.int64(50)
            self.activate_agents()
            assert type(self.agents[0].energy) is np.int64
            everyone = self.agents.where(np.ones(96, dtype=bool))
            assert everyone.age[0] == 11

    result = Model({"steps": 1}).gpu().run()
    assert result.info["auto_batch"]["batches"] == 1
    assert result.agents["age"][0] == 11


class _Wealth(am.Model):
    """Canonical quickstart idiom — matches docs/quickstart.rst vectorized lane."""

    def setup(self):
        n = int(self.p["n"])
        self.add_agents(n, wealth=np.ones(n, dtype=np.int64))

    def step(self):
        donors = self.agents.where(self.agents.wealth > 0)
        if len(donors) == 0:
            return
        donors.wealth -= 1
        ids = self.agents.ids.to_numpy()
        recipients = self.rng.choice(ids, size=len(donors))
        self.agents.at[recipients].scatter_add(wealth=1)


def test_run_cpu():
    res = _Wealth({"n": 200, "steps": 10, "seed": 0}).cpu().run()
    assert res.agents.height == 200
    assert int(res.agents["wealth"].sum()) == 200


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA/CuPy not available")
def test_run_gpu_same_step():
    res = _Wealth({"n": 200, "steps": 10, "seed": 0}).gpu().run()
    assert res.agents.height == 200
    assert int(res.agents["wealth"].sum()) == 200


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA/CuPy not available")
def test_cpu_gpu_conserves_wealth():
    """Device RNG differs from NumPy; both backends preserve total wealth."""
    params = {"n": 500, "steps": 25, "seed": 42}
    cpu = _Wealth(params).cpu().run()
    gpu = _Wealth(params).gpu().run()
    assert int(cpu.agents["wealth"].sum()) == 500
    assert int(gpu.agents["wealth"].sum()) == 500


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA/CuPy not available")
def test_recipient_scatter_avoids_host_round_trip():
    """Recipient ``scatter_add`` stays on device (fused debit/credit tested separately)."""
    from unittest.mock import patch

    m = _Wealth({"n": 100, "steps": 1, "seed": 0})
    m.setup()
    begin_execution(m, ExecutionConfig(device="gpu", mode="vectorized"))
    donors = m.agents.where(m.agents.wealth > 0)
    donors.wealth -= 1
    ids = m.agents.ids.to_numpy()
    recipients = m.rng.choice(ids, size=len(donors))

    def _forbid_export(arr):
        raise AssertionError("unexpected host export during GPU scatter_add")

    try:
        with patch("ambr.device_columns.to_host", side_effect=_forbid_export):
            with patch("ambr.gpu.to_host", side_effect=_forbid_export):
                m.agents.at[recipients].scatter_add(wealth=1)
    finally:
        end_execution(m)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA/CuPy not available")
def test_device_column_array_is_device_resident():
    m = _Wealth({"n": 50, "steps": 1, "seed": 0})
    m.setup()
    begin_execution(m, ExecutionConfig(device="gpu", mode="vectorized"))
    try:
        wealth = m.agents.wealth
        assert hasattr(wealth, "array")
        arr = wealth.array
        assert arr is get_device_column(m, "wealth")
        m.step()
        dev = to_host(get_device_column(m, "wealth"))
        stale = m.population.data["wealth"].to_numpy()
        assert int(dev.sum()) == 50
        assert np.array_equal(stale, np.ones(50))
        assert not np.array_equal(dev, stale)
    finally:
        end_execution(m)


def test_run_gpu_requires_cuda():
    if GPU_AVAILABLE:
        pytest.skip("GPU present")
    with pytest.raises(RuntimeError, match="GPU requested"):
        _Wealth({"n": 10, "steps": 1}).gpu().run()


def _vectorized_benchmark_models():
    """Same classes the AMBER (GPU) benchmark row uses."""
    import sys
    from pathlib import Path

    bench = Path(__file__).resolve().parents[1] / "benchmarks"
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    from models.amber_models import AMBER_VECTORIZED_MODELS

    return AMBER_VECTORIZED_MODELS


def test_benchmark_wealth_vectorized_lane_is_one_clean_reduction():
    cls = _vectorized_benchmark_models()["wealth_transfer"]
    res = cls({
        "n": 200,
        "steps": 5,
        "show_progress": False,
        "seed": 7,
        "initial_wealth": 1,
    }).cpu(mode="vectorized").run(contract="check")

    assert int(res.agents["wealth"].sum()) == 200
    assert int(res.agents["wealth"].min()) >= 0
    assert all(cert.clean for cert in res["contract"])


def test_benchmark_wealth_oop_freezes_donors_before_writes():
    """A recipient that starts at zero must not become a donor this step."""
    cls = _vectorized_benchmark_models()["wealth_transfer"]
    model = cls({
        "n": 2,
        "steps": 1,
        "show_progress": False,
        "seed": 0,
        "initial_wealth": 1,
    }).cpu(mode="oop")
    model._ensure_setup()
    agents = list(model.agents)
    agents[0].wealth = 1
    agents[1].wealth = 0
    model._flush_pending_writes()

    class _ChooseSecond:
        @staticmethod
        def choice(_agents):
            return agents[1]

    model.random = _ChooseSecond()
    res = model.run(contract="check")

    assert res.agents.sort("id")["wealth"].to_list() == [0, 1]
    assert res["contract"][0].clean


@pytest.mark.parametrize(
    "name",
    ["wealth_transfer", "random_walk", "sir_epidemic", "schelling"],
)
def test_vectorized_models_run_cpu(name):
    cls = _vectorized_benchmark_models()[name]
    cfg = {
        "n": 40,
        "steps": 3,
        "show_progress": False,
        "seed": 0,
        "initial_wealth": 1,
        "initial_infected": 3,
        "world_size": 50,
        "speed": 1.0,
        "movement_speed": 2.0,
        "infection_radius": 5.0,
        "transmission_rate": 0.1,
        "recovery_time": 14,
        "density": 0.8,
        "fraction_a": 0.5,
        "tolerance": 0.3,
    }
    res = cls(cfg).cpu().run()
    assert res.agents.height == 40


@pytest.mark.parametrize(
    "name",
    ["wealth_transfer", "random_walk", "sir_epidemic", "schelling"],
)
def test_vectorized_models_run_oop(name):
    """The same benchmark families expose a real CPU Agent-object lane."""
    cls = _vectorized_benchmark_models()[name]
    cfg = {
        "n": 40,
        "steps": 3,
        "show_progress": False,
        "seed": 0,
        "initial_wealth": 1,
        "initial_infected": 3,
        "world_size": 50,
        "speed": 1.0,
        "movement_speed": 2.0,
        "infection_radius": 5.0,
        "transmission_rate": 0.1,
        "recovery_time": 14,
        "density": 0.8,
        "fraction_a": 0.5,
        "tolerance": 0.3,
    }
    res = cls(cfg).cpu(mode="oop").run()
    assert res.agents.height == 40
    assert res.info.get("mode") == "oop"


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CUDA/CuPy not available")
@pytest.mark.parametrize(
    "name",
    ["wealth_transfer", "random_walk", "sir_epidemic", "schelling"],
)
def test_vectorized_models_run_gpu_same_api(name):
    """AMBER (GPU) must be model.gpu().run() on the same vectorized classes."""
    cls = _vectorized_benchmark_models()[name]
    cfg = {
        "n": 40,
        "steps": 3,
        "show_progress": False,
        "seed": 0,
        "initial_wealth": 1,
        "initial_infected": 3,
        "world_size": 50,
        "speed": 1.0,
        "movement_speed": 2.0,
        "infection_radius": 5.0,
        "transmission_rate": 0.1,
        "recovery_time": 14,
        "density": 0.8,
        "fraction_a": 0.5,
        "tolerance": 0.3,
    }
    res = cls(cfg).approve_fast_path(
        "workload smoke/invariant test"
    ).gpu().run()
    assert res.agents.height == 40
    assert res.info.get("device") == "gpu"

"""Per-framework well-mixed SIR implementations.

Each ``*_sir_curve(beta, gamma, n, steps, seed)`` runs the identical SIR on that
framework's engine and returns the infected-fraction curve. The common optimiser
(task.run_common_optimizer) drives the sequential frameworks; AMBER's GPU
ensemble has its own batched runner that evaluates every candidate at once.
"""

import os
import subprocess
import tempfile
import time

import numpy as np

I0 = 0.02
JULIA_BIN = os.environ.get("JULIA_BIN", os.path.expanduser("~/julia/bin/julia"))


def _init_status(n):
    status = np.zeros(n, dtype=np.int64)
    status[: max(1, int(I0 * n))] = 1
    return status


# --- AMBER (columnar CPU) --------------------------------------------------

def amber_sir_curve(beta, gamma, n, steps, seed):
    import ambr as am

    class _SIR(am.Model):
        def setup(self):
            self.add_agents(n, status=_init_status(n))

        def step(self):
            status = self.agents.numpy("status")
            i_frac = float((status == 1).mean())
            r1 = self.rng.random(len(status))
            r2 = self.rng.random(len(status))
            new = status.copy()
            new[(status == 0) & (r1 < beta * i_frac)] = 1
            new[(status == 1) & (r2 < gamma)] = 2
            self.agents.set(status=new)

        def update(self):
            self.record_model("i", float((self.agents.numpy("status") == 1).mean()))

    m = _SIR({"seed": seed, "steps": steps, "show_progress": False})
    return m.run()["model"]["i"].to_numpy()


# --- mesa-frames (AgentSetPolars) ------------------------------------------

def mesa_frames_sir_curve(beta, gamma, n, steps, seed):
    import polars as pl
    from mesa_frames import AgentSetPolars, ModelDF

    class _Set(AgentSetPolars):
        def __init__(self, model):
            super().__init__(model)
            self.add(pl.DataFrame({"unique_id": np.arange(n, dtype=np.int64),
                                   "status": _init_status(n)}))

        def step(self):
            status = self.agents["status"].to_numpy()
            i_frac = float((status == 1).mean())
            rng = self.model._rng
            r1 = rng.random(len(status))
            r2 = rng.random(len(status))
            new = status.copy()
            new[(status == 0) & (r1 < beta * i_frac)] = 1
            new[(status == 1) & (r2 < gamma)] = 2
            self.set({"status": new})

    class _M(ModelDF):
        def __init__(self):
            super().__init__()
            self._rng = np.random.default_rng(seed)
            self.s = _Set(self)
            self.curve = []

        def step(self):
            self.s.step()
            self.curve.append(float((self.s.agents["status"].to_numpy() == 1).mean()))

    m = _M()
    for _ in range(steps):
        m.step()
    return np.array(m.curve)


# --- Mesa (object-oriented, mesa 3.x) --------------------------------------

def mesa_sir_curve(beta, gamma, n, steps, seed):
    import mesa

    class _A(mesa.Agent):
        def __init__(self, model, status):
            super().__init__(model)
            self.status = status
            self.nxt = status

    class _M(mesa.Model):
        def __init__(self):
            super().__init__(seed=seed)
            st = _init_status(n)
            for s in st:
                _A(self, int(s))

        def step(self):
            agents = list(self.agents)
            i_frac = sum(a.status == 1 for a in agents) / len(agents)
            for a in agents:
                if a.status == 0 and self.random.random() < beta * i_frac:
                    a.nxt = 1
                elif a.status == 1 and self.random.random() < gamma:
                    a.nxt = 2
                else:
                    a.nxt = a.status
            for a in agents:
                a.status = a.nxt

    m = _M()
    curve = []
    for _ in range(steps):
        m.step()
        curve.append(sum(a.status == 1 for a in m.agents) / n)
    return np.array(curve)


# --- agentpy (object AgentList) --------------------------------------------

def agentpy_sir_curve(beta, gamma, n, steps, seed):
    import agentpy as ap

    class _M(ap.Model):
        def setup(self):
            self.pop = ap.AgentList(self, n)
            for a in self.pop:
                a.status = 0
            for a in self.pop[: max(1, int(I0 * n))]:
                a.status = 1
            self.curve = []

        def step(self):
            status = np.fromiter((a.status for a in self.pop), dtype=np.int64, count=n)
            i_frac = float((status == 1).mean())
            r1 = self.nprandom.random(n)
            r2 = self.nprandom.random(n)
            new = status.copy()
            new[(status == 0) & (r1 < beta * i_frac)] = 1
            new[(status == 1) & (r2 < gamma)] = 2
            for a, s in zip(self.pop, new):
                a.status = int(s)
            self.curve.append(float((new == 1).mean()))

    m = _M({"seed": seed, "steps": steps})
    m.run(steps=steps, display=False)
    return np.array(m.curve)


# --- FLAME GPU 2 (RTC CUDA, one GPU simulation per candidate) ---------------

_FLAMEGPU_WMSIR = r"""
FLAMEGPU_AGENT_FUNCTION(wmsir_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    int status = FLAMEGPU->getVariable<int>("status");
    float ifrac = FLAMEGPU->environment.getProperty<float>("i_frac");
    float beta  = FLAMEGPU->environment.getProperty<float>("beta");
    float gamma = FLAMEGPU->environment.getProperty<float>("gamma");
    if (status == 0) {
        if (FLAMEGPU->random.uniform<float>() < beta * ifrac)
            FLAMEGPU->setVariable<int>("status", 1);
    } else if (status == 1) {
        if (FLAMEGPU->random.uniform<float>() < gamma)
            FLAMEGPU->setVariable<int>("status", 2);
    }
    return flamegpu::ALIVE;
}
"""


def flamegpu_sir_curve(beta, gamma, n, steps, seed):
    import pyflamegpu

    curve = []

    class _Rec(pyflamegpu.HostFunction):
        def run(self, FLAMEGPU):
            frac = FLAMEGPU.agent("person").countInt("status", 1) / n
            curve.append(frac)
            FLAMEGPU.environment.setPropertyFloat("i_frac", float(frac))

    m = pyflamegpu.ModelDescription("wmsir")
    env = m.Environment()
    env.newPropertyFloat("i_frac", I0)
    env.newPropertyFloat("beta", float(beta))
    env.newPropertyFloat("gamma", float(gamma))
    a = m.newAgent("person")
    a.newVariableInt("status", 0)
    m.newLayer().addAgentFunction(a.newRTCFunction("wmsir_step", _FLAMEGPU_WMSIR))
    m.addStepFunction(_Rec())

    k = max(1, int(I0 * n))
    pop = pyflamegpu.AgentVector(a, n)   # status defaults to 0 (S)
    for i in range(k):                   # only the initially-infected need setting
        pop[i].setVariableInt("status", 1)

    sim = pyflamegpu.CUDASimulation(m)
    sim.SimulationConfig().steps = steps
    sim.SimulationConfig().random_seed = seed
    sim.setPopulationData(pop)
    sim.simulate()
    return np.array(curve)


# --- AMBER GPU batched ensemble (all candidates in one pass) ---------------

def amber_gpu_calibrate(observed, candidates, n, steps, eval_seed, batch=256):
    from ambr.gpu import get_array_module, to_host
    from ambr.gpu_ensemble import BatchedWellMixedSIR, GPUEnsembleRunner

    runner = GPUEnsembleRunner(BatchedWellMixedSIR())
    betas = np.array([c[0] for c in candidates], dtype=np.float64)
    gammas = np.array([c[1] for c in candidates], dtype=np.float64)
    B = len(candidates)
    xp = get_array_module()
    obs = xp.asarray(observed, dtype=xp.float32).reshape(1, -1)

    t0 = time.perf_counter()
    losses = np.empty(B, dtype=np.float64)
    for s in range(0, B, batch):
        e = min(s + batch, B)
        params = {"beta": betas[s:e], "gamma": gammas[s:e],
                  "i0_frac": np.full(e - s, I0)}
        traj = runner.run(n, steps, params, seed=eval_seed)
        losses[s:e] = to_host(((traj["I_frac"] - obs) ** 2).sum(axis=1))
    wall = time.perf_counter() - t0

    best = int(np.argmin(losses))
    return {
        "theta_hat": {"beta": float(betas[best]), "gamma": float(gammas[best])},
        "best_loss": float(losses[best]), "n_evals": B, "wall_s": wall,
        "evals_per_s": B / max(wall, 1e-9), "curve": None,
    }


CURVE_FNS = {
    "AMBER": amber_sir_curve,
    "mesa-frames": mesa_frames_sir_curve,
    "Mesa": mesa_sir_curve,
    "agentpy": agentpy_sir_curve,
}


# --- Agents.jl (Julia subprocess, self-contained calibration) --------------

def agentsjl_calibrate(observed, candidates, n, steps, eval_seed, val_seeds, gt, bounds):
    """Run the whole calibration inside one Julia process (JIT warmed, then timed)."""
    script = os.path.join(os.path.dirname(__file__), "agentsjl_sir_calib.jl")
    fd, prob = tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd, "w") as f:
        f.write(f"{n} {steps} {eval_seed}\n")
        f.write(" ".join(f"{v:.10f}" for v in observed) + "\n")
        f.write(" ".join(str(s) for s in val_seeds) + "\n")
        f.write(f"{gt['beta']} {gt['gamma']}\n")
        f.write(f"{bounds['beta'][0]} {bounds['beta'][1]} "
                f"{bounds['gamma'][0]} {bounds['gamma'][1]}\n")
        for (b, g) in candidates:
            f.write(f"{b:.10f} {g:.10f}\n")
    try:
        out = subprocess.run([JULIA_BIN, script, prob],
                             capture_output=True, text=True, timeout=900)
    finally:
        os.unlink(prob)
    if out.returncode != 0:
        raise RuntimeError(f"julia exited {out.returncode}: {out.stderr[-400:]}")
    beta, gamma, best_loss, n_evals, wall, val, rec = \
        [float(x) for x in out.stdout.strip().splitlines()[-1].split()]
    return {"theta_hat": {"beta": beta, "gamma": gamma}, "best_loss": best_loss,
            "n_evals": int(n_evals), "wall_s": wall,
            "evals_per_s": int(n_evals) / max(wall, 1e-9),
            "val_loss": val, "recovery_error": rec}

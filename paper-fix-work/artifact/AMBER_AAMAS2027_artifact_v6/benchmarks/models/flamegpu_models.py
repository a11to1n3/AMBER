"""FLAME GPU 2 benchmark models (pyflamegpu, RTC agent functions).

Implements the four benchmark models on FLAME GPU 2 with runtime-compiled
CUDA-C agent functions. Most models initialise agents **on the GPU** (variable
defaults + a step-0 branch inside the first agent function) so there is no
per-agent Python loop -- this keeps the framework usable up to N = 1e6+.

* random_walk     -- parallel move, no messaging.
* sir_epidemic    -- spatial messaging (MessageSpatial2D): infected broadcast
                     position+status; susceptibles read neighbours in radius.
* wealth_transfer -- bucket messaging (MessageBucket): each agent with wealth
                     sends 1 to a random recipient (keyed by agent ID).
* schelling       -- toroidal grid segregation (MessageArray2D Moore wrap +
                     macro-property occupancy grid for relocation).

Requires a CUDA toolkit on the host (CUDA_PATH) matching the wheel's bundled
NVRTC so RTC can compile the agent functions.
"""

import math

import pyflamegpu

SEED = 42


# --------------------------------------------------------------------------- #
# Random Walk
# --------------------------------------------------------------------------- #

_RW_MOVE = r"""
FLAMEGPU_AGENT_FUNCTION(rw_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float ws = FLAMEGPU->environment.getProperty<float>("world_size");
    float x, y;
    if (FLAMEGPU->getStepCounter() == 0) {           // GPU-side init
        x = FLAMEGPU->random.uniform<float>(0.0f, ws);
        y = FLAMEGPU->random.uniform<float>(0.0f, ws);
    } else {
        x = FLAMEGPU->getVariable<float>("x");
        y = FLAMEGPU->getVariable<float>("y");
    }
    const float speed = FLAMEGPU->environment.getProperty<float>("speed");
    x += FLAMEGPU->random.uniform<float>(-speed, speed);
    y += FLAMEGPU->random.uniform<float>(-speed, speed);
    x = fminf(fmaxf(x, 0.0f), ws);
    y = fminf(fmaxf(y, 0.0f), ws);
    FLAMEGPU->setVariable<float>("x", x);
    FLAMEGPU->setVariable<float>("y", y);
    return flamegpu::ALIVE;
}
"""


class WalkModel:
    def __init__(self, n, steps, cfg):
        self.n, self.steps = n, steps
        m = pyflamegpu.ModelDescription("rw")
        env = m.Environment()
        env.newPropertyFloat("world_size", float(cfg.get("world_size", 100)))
        env.newPropertyFloat("speed", float(cfg.get("speed", 1.0)))
        a = m.newAgent("walker")
        a.newVariableFloat("x", 0.0)
        a.newVariableFloat("y", 0.0)
        m.newLayer().addAgentFunction(a.newRTCFunction("rw_move", _RW_MOVE))
        self.model, self.agent = m, a

    def run(self):
        pop = pyflamegpu.AgentVector(self.agent, self.n)  # bulk default init (C++)
        sim = pyflamegpu.CUDASimulation(self.model)
        sim.SimulationConfig().steps = self.steps
        sim.SimulationConfig().random_seed = SEED
        sim.setPopulationData(pop)
        sim.simulate()


# --------------------------------------------------------------------------- #
# SIR Epidemic (spatial messaging)
# --------------------------------------------------------------------------- #

_SIR_MOVE_OUT = r"""
FLAMEGPU_AGENT_FUNCTION(sir_move_out, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    const float ws = FLAMEGPU->environment.getProperty<float>("world_size");
    float x, y;
    int status;
    if (FLAMEGPU->getStepCounter() == 0) {           // GPU-side init
        x = FLAMEGPU->random.uniform<float>(0.0f, ws);
        y = FLAMEGPU->random.uniform<float>(0.0f, ws);
        const unsigned int ii = FLAMEGPU->environment.getProperty<unsigned int>("initial_infected");
        status = (FLAMEGPU->getID() <= ii) ? 1 : 0;
        FLAMEGPU->setVariable<int>("status", status);
    } else {
        x = FLAMEGPU->getVariable<float>("x");
        y = FLAMEGPU->getVariable<float>("y");
        status = FLAMEGPU->getVariable<int>("status");
    }
    const float speed = FLAMEGPU->environment.getProperty<float>("speed");
    x += FLAMEGPU->random.uniform<float>(-speed, speed);
    y += FLAMEGPU->random.uniform<float>(-speed, speed);
    x = fminf(fmaxf(x, 0.0f), ws);
    y = fminf(fmaxf(y, 0.0f), ws);
    FLAMEGPU->setVariable<float>("x", x);
    FLAMEGPU->setVariable<float>("y", y);
    FLAMEGPU->message_out.setVariable<int>("status", status);
    FLAMEGPU->message_out.setLocation(x, y);
    return flamegpu::ALIVE;
}
"""

_SIR_INFECT = r"""
FLAMEGPU_AGENT_FUNCTION(sir_infect, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("status") != 0) return flamegpu::ALIVE;  // susceptible only
    const float x = FLAMEGPU->getVariable<float>("x");
    const float y = FLAMEGPU->getVariable<float>("y");
    const float r2 = FLAMEGPU->environment.getProperty<float>("radius2");
    const float trans = FLAMEGPU->environment.getProperty<float>("transmission_rate");
    for (const auto& m : FLAMEGPU->message_in(x, y)) {
        if (m.getVariable<int>("status") == 1) {
            const float dx = m.getVariable<float>("x") - x;
            const float dy = m.getVariable<float>("y") - y;
            if (dx * dx + dy * dy <= r2 && FLAMEGPU->random.uniform<float>() < trans) {
                FLAMEGPU->setVariable<int>("status", 1);
                FLAMEGPU->setVariable<int>("infection_time", 0);
                return flamegpu::ALIVE;
            }
        }
    }
    return flamegpu::ALIVE;
}
"""

_SIR_RECOVER = r"""
FLAMEGPU_AGENT_FUNCTION(sir_recover, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("status") == 1) {
        const int t = FLAMEGPU->getVariable<int>("infection_time") + 1;
        if (t >= FLAMEGPU->environment.getProperty<int>("recovery_time")) {
            FLAMEGPU->setVariable<int>("status", 2);
        } else {
            FLAMEGPU->setVariable<int>("infection_time", t);
        }
    }
    return flamegpu::ALIVE;
}
"""


class SIRModel:
    def __init__(self, n, steps, cfg):
        self.n, self.steps = n, steps
        ws = float(cfg.get("world_size", 100))
        radius = float(cfg.get("infection_radius", 5.0))
        m = pyflamegpu.ModelDescription("sir")
        env = m.Environment()
        env.newPropertyFloat("world_size", ws)
        env.newPropertyFloat("speed", float(cfg.get("movement_speed", 2.0)))
        env.newPropertyFloat("radius2", radius * radius)
        env.newPropertyFloat("transmission_rate", float(cfg.get("transmission_rate", 0.1)))
        env.newPropertyInt("recovery_time", int(cfg.get("recovery_time", 14)))
        env.newPropertyUInt("initial_infected", int(cfg.get("initial_infected", 5)))
        a = m.newAgent("person")
        a.newVariableFloat("x", 0.0)
        a.newVariableFloat("y", 0.0)
        a.newVariableInt("status", 0)
        a.newVariableInt("infection_time", 0)
        msg = m.newMessageSpatial2D("loc")
        msg.setRadius(radius)
        msg.setMin(0.0, 0.0)
        msg.setMax(ws, ws)
        msg.newVariableInt("status")
        out = a.newRTCFunction("sir_move_out", _SIR_MOVE_OUT)
        out.setMessageOutput("loc")
        inf = a.newRTCFunction("sir_infect", _SIR_INFECT)
        inf.setMessageInput("loc")
        rec = a.newRTCFunction("sir_recover", _SIR_RECOVER)
        m.newLayer().addAgentFunction(out)
        m.newLayer().addAgentFunction(inf)
        m.newLayer().addAgentFunction(rec)
        self.model, self.agent = m, a

    def run(self):
        pop = pyflamegpu.AgentVector(self.agent, self.n)
        sim = pyflamegpu.CUDASimulation(self.model)
        sim.SimulationConfig().steps = self.steps
        sim.SimulationConfig().random_seed = SEED
        sim.setPopulationData(pop)
        sim.simulate()


# --------------------------------------------------------------------------- #
# Wealth Transfer (bucket messaging, keyed by agent ID)
# --------------------------------------------------------------------------- #

_WT_GIVE = r"""
FLAMEGPU_AGENT_FUNCTION(wt_give, flamegpu::MessageNone, flamegpu::MessageBucket) {
    const int w = FLAMEGPU->getVariable<int>("wealth");
    if (w > 0) {
        const unsigned int n = FLAMEGPU->environment.getProperty<unsigned int>("n");
        const unsigned int r = FLAMEGPU->random.uniform<unsigned int>(1u, n);
        FLAMEGPU->message_out.setKey(r);
        FLAMEGPU->message_out.setVariable<int>("amount", 1);
        FLAMEGPU->setVariable<int>("wealth", w - 1);
    }
    return flamegpu::ALIVE;
}
"""

_WT_RECEIVE = r"""
FLAMEGPU_AGENT_FUNCTION(wt_receive, flamegpu::MessageBucket, flamegpu::MessageNone) {
    int sum = 0;
    for (const auto& m : FLAMEGPU->message_in(FLAMEGPU->getID())) {
        sum += m.getVariable<int>("amount");
    }
    if (sum) {
        FLAMEGPU->setVariable<int>("wealth", FLAMEGPU->getVariable<int>("wealth") + sum);
    }
    return flamegpu::ALIVE;
}
"""


class WealthModel:
    def __init__(self, n, steps, cfg):
        self.n, self.steps = n, steps
        m = pyflamegpu.ModelDescription("wt")
        env = m.Environment()
        env.newPropertyUInt("n", n)
        a = m.newAgent("trader")
        a.newVariableInt("wealth", int(cfg.get("initial_wealth", 1)))  # default
        msg = m.newMessageBucket("transfer")
        msg.setBounds(1, n + 1)  # keyed by agent ID (1..n)
        msg.newVariableInt("amount")
        give = a.newRTCFunction("wt_give", _WT_GIVE)
        give.setMessageOutput("transfer")
        give.setMessageOutputOptional(True)
        recv = a.newRTCFunction("wt_receive", _WT_RECEIVE)
        recv.setMessageInput("transfer")
        m.newLayer().addAgentFunction(give)
        m.newLayer().addAgentFunction(recv)
        self.model, self.agent = m, a

    def run(self):
        pop = pyflamegpu.AgentVector(self.agent, self.n)  # bulk default wealth, no loop
        sim = pyflamegpu.CUDASimulation(self.model)
        sim.SimulationConfig().steps = self.steps
        sim.SimulationConfig().random_seed = SEED
        sim.setPopulationData(pop)
        sim.simulate()


# --------------------------------------------------------------------------- #
# Schelling Segregation (MessageArray2D + macro-property grid)
# --------------------------------------------------------------------------- #

class _SchellingUploadHost(pyflamegpu.HostFunction):
    """Copy host placement arrays into init macro-properties before step 0."""

    def __init__(self, owner):
        super().__init__()
        self._owner = owner

    def run(self, FLAMEGPU):
        xs = self._owner._pending_xs
        ys = self._owner._pending_ys
        ts = self._owner._pending_ts
        if xs is None:
            return
        mx = FLAMEGPU.environment.getMacroPropertyUInt("init_x")
        my = FLAMEGPU.environment.getMacroPropertyUInt("init_y")
        mt = FLAMEGPU.environment.getMacroPropertyInt("init_t")
        for i in range(xs.shape[0]):
            mx[i][0] = int(xs[i])
            my[i][0] = int(ys[i])
            mt[i][0] = int(ts[i])


def _schelling_rtc(G: int, n: int) -> tuple[str, str, str, str, str]:
    """Return RTC sources with grid dimension G and init-buffer length n baked in."""
    init = f"""
FLAMEGPU_AGENT_FUNCTION(sc_init, flamegpu::MessageNone, flamegpu::MessageNone) {{
    if (FLAMEGPU->getStepCounter() != 0) return flamegpu::ALIVE;
    const unsigned int idx = FLAMEGPU->getID() - 1u;
    auto init_x = FLAMEGPU->environment.getMacroProperty<unsigned int, {n}, 1>("init_x");
    auto init_y = FLAMEGPU->environment.getMacroProperty<unsigned int, {n}, 1>("init_y");
    auto init_t = FLAMEGPU->environment.getMacroProperty<int, {n}, 1>("init_t");
    const unsigned int x = init_x[idx];
    const unsigned int y = init_y[idx];
    const int atype = init_t[idx];
    FLAMEGPU->setVariable<unsigned int>("x", x);
    FLAMEGPU->setVariable<unsigned int>("y", y);
    FLAMEGPU->setVariable<int>("atype", atype);
    auto grid = FLAMEGPU->environment.getMacroProperty<int, {G}, {G}>("grid");
    grid[x][y].CAS(0, atype);
    return flamegpu::ALIVE;
}}
"""
    publish = r"""
FLAMEGPU_AGENT_FUNCTION(sc_publish, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    FLAMEGPU->message_out.setVariable<int>("atype", FLAMEGPU->getVariable<int>("atype"));
    FLAMEGPU->message_out.setIndex(
        FLAMEGPU->getVariable<unsigned int>("x"),
        FLAMEGPU->getVariable<unsigned int>("y"));
    return flamegpu::ALIVE;
}
"""
    eval_happy = r"""
FLAMEGPU_AGENT_FUNCTION(sc_eval, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    const unsigned int y = FLAMEGPU->getVariable<unsigned int>("y");
    const int my_type = FLAMEGPU->getVariable<int>("atype");
    const float tol = FLAMEGPU->environment.getProperty<float>("tolerance");
    int same = 0, total = 0;
    for (const auto& m : FLAMEGPU->message_in.wrap(x, y)) {
        const int t = m.getVariable<int>("atype");
        if (t != 0) {
            ++total;
            if (t == my_type) ++same;
        }
    }
    const int happy = (total == 0) || (static_cast<float>(same) >= tol * static_cast<float>(total));
    FLAMEGPU->setVariable<int>("happy", happy ? 1 : 0);
    return flamegpu::ALIVE;
}
"""
    move = f"""
FLAMEGPU_AGENT_FUNCTION(sc_move, flamegpu::MessageNone, flamegpu::MessageNone) {{
    if (FLAMEGPU->getVariable<int>("happy") != 0) return flamegpu::ALIVE;
    const unsigned int ox = FLAMEGPU->getVariable<unsigned int>("x");
    const unsigned int oy = FLAMEGPU->getVariable<unsigned int>("y");
    const int atype = FLAMEGPU->getVariable<int>("atype");
    const unsigned int G = FLAMEGPU->environment.getProperty<unsigned int>("grid_size");
    auto grid = FLAMEGPU->environment.getMacroProperty<int, {G}, {G}>("grid");
    const unsigned int max_attempts = 32u;
    for (unsigned int attempt = 0; attempt < max_attempts; ++attempt) {{
        const unsigned int nx = FLAMEGPU->random.uniform<unsigned int>(0u, G - 1u);
        const unsigned int ny = FLAMEGPU->random.uniform<unsigned int>(0u, G - 1u);
        if (grid[nx][ny].CAS(0, atype) == 0) {{
            grid[ox][oy].exchange(0);
            FLAMEGPU->setVariable<unsigned int>("x", nx);
            FLAMEGPU->setVariable<unsigned int>("y", ny);
            break;
        }}
    }}
    return flamegpu::ALIVE;
}}
"""
    return init, publish, eval_happy, move


class SchellingModel:
    """Schelling segregation on a toroidal G x G grid (matches _schelling_core params)."""

    def __init__(self, n, steps, cfg):
        self.n, self.steps = n, steps
        density = float(cfg.get("density", 0.8))
        self.G = int(math.ceil(math.sqrt(n / density)))
        init_src, publish_src, eval_src, move_src = _schelling_rtc(self.G, n)

        m = pyflamegpu.ModelDescription("schelling")
        env = m.Environment()
        env.newPropertyUInt("grid_size", self.G)
        env.newPropertyFloat("tolerance", float(cfg.get("tolerance", 0.3)))
        env.newMacroPropertyInt("grid", self.G, self.G)
        # 1 x n macro-properties: RTC can index these at N > 1e4 (env arrays cannot).
        env.newMacroPropertyUInt("init_x", n, 1)
        env.newMacroPropertyUInt("init_y", n, 1)
        env.newMacroPropertyInt("init_t", n, 1)

        msg = m.newMessageArray2D("cell")
        msg.setDimensions(self.G, self.G)
        msg.newVariableInt("atype")

        a = m.newAgent("resident")
        a.newVariableUInt("x", 0)
        a.newVariableUInt("y", 0)
        a.newVariableInt("atype", 1)
        a.newVariableInt("happy", 1)

        init_fn = a.newRTCFunction("sc_init", init_src)
        pub = a.newRTCFunction("sc_publish", publish_src)
        pub.setMessageOutput("cell")
        ev = a.newRTCFunction("sc_eval", eval_src)
        ev.setMessageInput("cell")
        mv = a.newRTCFunction("sc_move", move_src)

        m.addInitFunction(_SchellingUploadHost(self))
        m.newLayer().addAgentFunction(init_fn)
        m.newLayer().addAgentFunction(pub)
        m.newLayer().addAgentFunction(ev)
        m.newLayer().addAgentFunction(mv)
        self.model, self.agent = m, a
        self._density = density
        self._fraction_a = float(cfg.get("fraction_a", 0.5))
        self._pending_xs = None
        self._pending_ys = None
        self._pending_ts = None

    def run(self):
        import numpy as np

        from models._schelling_core import schelling_setup

        rng = np.random.default_rng(SEED)
        xs, ys, ts, _G = schelling_setup(
            self.n, self._density, self._fraction_a, rng, np
        )
        self._pending_xs = np.ascontiguousarray(xs, dtype=np.uint32)
        self._pending_ys = np.ascontiguousarray(ys, dtype=np.uint32)
        self._pending_ts = np.ascontiguousarray(ts, dtype=np.int32)

        pop = pyflamegpu.AgentVector(self.agent, self.n)
        sim = pyflamegpu.CUDASimulation(self.model)
        sim.SimulationConfig().steps = self.steps
        sim.SimulationConfig().random_seed = SEED
        sim.setPopulationData(pop)
        sim.simulate()


FLAMEGPU_MODELS = {
    "wealth_transfer": WealthModel,
    "random_walk": WalkModel,
    "sir_epidemic": SIRModel,
    "schelling": SchellingModel,
}


if __name__ == "__main__":
    import time
    cfgs = {
        "random_walk": {"world_size": 100, "speed": 1.0},
        "sir_epidemic": {"initial_infected": 5, "world_size": 100, "movement_speed": 2.0,
                          "infection_radius": 5.0, "transmission_rate": 0.1, "recovery_time": 14},
        "wealth_transfer": {"initial_wealth": 1},
        "schelling": {"density": 0.8, "fraction_a": 0.5, "tolerance": 0.3},
    }
    for name, cls in FLAMEGPU_MODELS.items():
        try:
            cls(1000, 5, cfgs[name]).run()  # warm up / compile
            t0 = time.perf_counter()
            cls(1_000_000, 50, cfgs[name]).run()
            print(f"  {name:16s} OK  (1e6 agents, 50 steps: {time.perf_counter()-t0:.2f}s)")
        except Exception as e:
            print(f"  {name:16s} FAIL: {type(e).__name__}: {str(e)[:300]}")

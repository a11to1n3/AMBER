#!/usr/bin/env python
"""
theorem_referee.py — executable referee for the AMBER execution-semantics theorems.

Builds a minimal faithful model of the two semantics of Section 3.2 and
stress-tests the paper's results by exhaustive/randomized counterexample search.

Semantics (Section 3.2), on cells (agent, column):
  - Snap(W)  : every event's value function reads the STEP-ENTRY state; a cell's
               result is (a) the single set value, (b) the AC fold of same-operator
               reduces, or (c) CONFLICT (>=2 sets, set+reduce mix, mixed/absent AC).
  - Sched_pi(W): events applied one at a time in order pi, each reading the RUNNING
               state (sequential activation).

Non-interference (Definition 3.1):
  (I)  no event writes a cell another event reads (no read-after-write), and
  (II) every shared-target cell receives only reduces with a common AC operator.

Results checked:
  Thm 3.1 (sufficient direction)  Prop 3.1 (AC folds)
  Thm 3.2 (confluence)  Thm 3.3 (one-barrier construction)

Findings (see REVIEW_REPORT.md and the generated JSON):
  * No generated non-interfering multiset violates the sufficient theorem.
  * Some interfering multisets have no divergence witness in the finite grid,
    so the experiment does not support a converse; the paper makes none.
  * Non-AC-but-order-safe folds (for example subtraction translations) show
    why the AC rule is sufficient and conservative.
"""
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "theorem_referee_results.json"

CELLS = [(i, 'c') for i in range(1, 5)]

# ---------- the two semantics ----------
def snap(state, events):
    by = {}
    for e in events:
        by.setdefault(e['tgt'], []).append(e)
    out = dict(state)
    for tgt, evs in by.items():
        if len(evs) == 1 and evs[0]['mode'] == 'set':
            out[tgt] = evs[0]['fn'](state)
        elif len(evs) == 1 and evs[0]['mode'] == 'reduce':
            out[tgt] = evs[0]['op'](state[tgt], evs[0]['fn'](state))           # single reduce: always defined
        elif (all(e['mode'] == 'reduce' for e in evs)
              and len({e.get('acname') for e in evs}) == 1 and evs[0].get('ac')):
            acc = state[tgt]
            for e in evs:                                                       # >=2 reduces: need same AC op
                acc = e['op'](acc, e['fn'](state))
            out[tgt] = acc
        else:
            out[tgt] = 'CONFLICT'                                              # >=2 sets / mix / non-AC
    return out

def sched(state, events, order):
    M = dict(state)
    for idx in order:
        e = events[idx]; v = e['fn'](M)                                        # reads RUNNING state
        M[e['tgt']] = v if e['mode'] == 'set' else e['op'](M[e['tgt']], v)
    return M

def is_noninterfering(events):
    for i, w in enumerate(events):
        for j, w2 in enumerate(events):
            if i == j:
                continue
            if w['tgt'] in w2['rd']:                                            # (I)
                return False
            if w['tgt'] == w2['tgt'] and not (                                  # (II)
                w['mode'] == 'reduce' and w2['mode'] == 'reduce'
                and w.get('acname') == w2.get('acname') and w.get('ac')):
                return False
    return True

# ---------- random event generation respecting tightness (rd == actual reads) ----------
AC_OPS = {'add': (lambda a, b: a + b, True), 'mul': (lambda a, b: a * b, True),
          'max': (lambda a, b: max(a, b), True)}
NONAC  = {'sub2': (lambda a, b: 2 * a + b, False)}   # genuinely order-dependent
ALL_OPS = {**AC_OPS, **NONAC}

def rand_event(rng):
    tgt = rng.choice(CELLS)
    rd = set(rng.sample(CELLS, rng.randint(0, 2)))
    src = sorted(rd)
    base = rng.randint(1, 3)
    fn = (lambda S, src=src, base=base: sum((k + 1) * S[c] for k, c in enumerate(src)) + base)
    if rng.random() < 0.5:
        return {'tgt': tgt, 'mode': 'set', 'rd': rd, 'fn': fn}
    opn = rng.choice(list(ALL_OPS)); op, ac = ALL_OPS[opn]
    return {'tgt': tgt, 'mode': 'reduce', 'op': op, 'acname': opn, 'ac': ac, 'rd': rd, 'fn': fn}

def diverges_exhaustive(events, domain=range(5)):
    for vals in itertools.product(domain, repeat=len(CELLS)):
        S = dict(zip(CELLS, vals)); sn = snap(S, events)
        if any(v == 'CONFLICT' for v in sn.values()):
            return True
        outs = {tuple(sorted(sched(S, events, list(o)).items()))
                for o in itertools.permutations(range(len(events)))}
        if len(outs) > 1 or tuple(sorted(sn.items())) not in outs:
            return True
    return False

def value_preserving_on_interfering_pair(events, domain=range(5)):
    """Check whether sampled condition-(I) writers preserve values on this grid."""
    for i, w in enumerate(events):
        for j, w2 in enumerate(events):
            if i == j:
                continue
            if w['tgt'] in w2['rd']:
                for vals in itertools.product(domain, repeat=len(CELLS)):
                    S = dict(zip(CELLS, vals))
                    newv = (w['op'](S[w['tgt']], w['fn'](S)) if w['mode'] == 'reduce'
                            else w['fn'](S))
                    if newv != S[w['tgt']]:
                        return False
    return True

# ---------- Theorem 3.1 ----------
def check_thm31(trials=8000, seed=7):
    """Exercise the sufficient direction and retain former-converse controls."""
    rng = random.Random(seed)
    sound_v = ni_ok = no_witness = int_c = 0
    for _ in range(trials):
        ev = [rand_event(rng) for _ in range(rng.randint(2, 4))]
        Ss = [{c: rng.randint(0, 9) for c in CELLS} for _ in range(16)]
        ni = is_noninterfering(ev)
        lhs = True
        for S in Ss:
            sn = snap(S, ev)
            if any(v == 'CONFLICT' for v in sn.values()):
                lhs = False; break
            if any(sched(S, ev, list(o)) != sn
                   for o in itertools.permutations(range(len(ev)))):
                lhs = False; break
        if ni:
            ni_ok += lhs; sound_v += (not lhs)                                 # (<=) must never fail
        else:
            int_c += 1
            if not diverges_exhaustive(ev):
                no_witness += 1
    return dict(non_interfering_ok=ni_ok, soundness_violations=sound_v,
                interfering=int_c, interfering_no_divergence_witness=no_witness)

def audit_no_witness(trials=8000, seed=7):
    """Describe the no-witness cases produced by this finite generator."""
    rng = random.Random(seed)
    caps = []
    for _ in range(trials):
        ev = [rand_event(rng) for _ in range(rng.randint(2, 4))]
        _ = [{c: rng.randint(0, 9) for c in CELLS} for _ in range(16)]         # keep stream in sync
        if not is_noninterfering(ev) and not diverges_exhaustive(ev):
            caps.append(ev)
    allmask = all(value_preserving_on_interfering_pair(ev) for ev in caps)
    return dict(no_witness=len(caps), all_value_preserving=allmask)

# ---------- Theorem 3.2 (confluence) ----------
def clean_rule(S):
    ev  = [{'tgt': (i, 'c'), 'mode': 'set', 'rd': {(i, 'c')},
            'fn': (lambda St, i=i: St[(i, 'c')] + 1)} for i in range(1, 4)]
    ev += [{'tgt': (4, 'd'), 'mode': 'reduce', 'op': lambda a, b: a + b,
            'acname': 'add', 'ac': True, 'rd': {(i, 'd')},
            'fn': (lambda St, i=i: St[(i, 'd')])} for i in range(1, 4)]
    return ev

def dirty_rule(S):
    return [{'tgt': (i, 'c'), 'mode': 'set', 'rd': {((i - 2) % 4 + 1, 'c')},
             'fn': (lambda St, i=i: St[((i - 2) % 4 + 1, 'c')] + 1)} for i in range(1, 5)]

def check_thm32(seed=11, runs=300, scheds=20, T=5):
    rng = random.Random(seed)
    grid = [(i, 'c') for i in range(1, 5)] + [(i, 'd') for i in range(1, 5)]
    mism = tot = 0
    for _ in range(runs):
        S0 = {c: rng.randint(0, 5) for c in grid}
        S = dict(S0)
        for _ in range(T):
            S = snap(S, clean_rule(S))
        if any(v == 'CONFLICT' for v in S.values()):
            continue
        n = len(clean_rule(S0))
        for _ in range(scheds):
            M = dict(S0)
            for t in range(T):
                M = sched(M, clean_rule(M), rng.sample(range(n), n))
            tot += 1; mism += (M != S)
    # Negative control only: this selected interfering rule is non-confluent.
    # The theorem does not claim that every interfering rule must diverge.
    S0 = {c: rng.randint(1, 9) for c in grid}
    ends = set()
    for _ in range(200):
        M = dict(S0)
        for _ in range(3):
            M = sched(M, dirty_rule(M), rng.sample(range(4), 4))
        ends.add(tuple(sorted((k, v) for k, v in M.items() if k[1] == 'c')))
    return dict(clean_runs=tot, mismatches=mism, dirty_distinct_endpoints=len(ends))

# ---------- Theorem 3.3 (one-barrier sufficient construction) ----------
def check_thm33(N=6):
    def fused(S):
        return [{'tgt': (i, 'c'), 'mode': 'set', 'rd': {((i - 2) % N + 1, 'c')},
                 'fn': (lambda St, i=i: St[((i - 2) % N + 1, 'c')])} for i in range(1, N + 1)]
    grid = [(i, 'c') for i in range(1, N + 1)]
    S0 = {(i, 'c'): i for i in range(1, N + 1)}
    sn = snap(S0, fused(S0))
    sc = sched(S0, fused(S0), list(range(N)))
    wrong0 = sum(1 for c in grid if sn[c] != sc[c])
    # 1 barrier: c'[i]=nbr; copy c=c'  -> equals snapshot for ALL orders
    P0 = [{'tgt': (i, 'c2'), 'mode': 'set', 'rd': {((i - 2) % N + 1, 'c')},
           'fn': (lambda St, i=i: St[((i - 2) % N + 1, 'c')])} for i in range(1, N + 1)]
    m = dict(S0); [m.setdefault((i, 'c2'), 0) for i in range(1, N + 1)]
    staged_ok = True
    for order in itertools.permutations(range(N)):
        mm = sched(m, P0, list(order))
        if any(mm[(i, 'c2')] != sn[(i, 'c')] for i in range(1, N + 1)):
            staged_ok = False; break
    return dict(zero_barrier_wrong_cells=wrong0, n_cells=N, one_barrier_exact_all_orders=staged_ok)


# ---------- Proposition A.1 (false positive: subtraction) ----------
def check_propA1(seed=5, states=2000):
    """A subtraction reduce family is flagged interfering (non-AC) yet its scheduled
    left-fold is order-invariant on every tested state — the characterized false positive."""
    rng = random.Random(seed)
    sub = [{'tgt': (1, 'c'), 'mode': 'reduce', 'op': (lambda a, b: a - b),
            'acname': 'sub', 'ac': False, 'rd': set(), 'fn': (lambda S, d=d: d)}
           for d in (2, 5, 3)]
    flagged = not is_noninterfering(sub)
    order_invariant = True
    for _ in range(states):
        S = {(1, 'c'): rng.randint(-50, 50)}
        outs = {sched(S, sub, list(o))[(1, 'c')]
                for o in itertools.permutations(range(len(sub)))}
        if len(outs) != 1:
            order_invariant = False; break
    return dict(flagged_interfering=flagged, scheduled_order_invariant=order_invariant)

if __name__ == '__main__':
    results = {
        "scope": (
            "Bounded regression checks for the sufficient theorem, trajectory "
            "confluence under its premise, the one-barrier construction, and "
            "commuting non-AC updates; not a proof or necessity test."
        ),
        "theorem_3_1_sufficient": check_thm31(),
        "former_converse_control": audit_no_witness(),
        "theorem_3_2_confluence": check_thm32(),
        "theorem_3_3_one_barrier": check_thm33(),
        "commuting_non_ac_control": check_propA1(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"Wrote {OUTPUT}")

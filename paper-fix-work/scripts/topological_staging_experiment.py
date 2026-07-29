"""Bounded experiment for sequential-topological staged semantics.

This is deliberately not a theorem about global step-entry snapshot semantics
and not a proof of a general barrier lower bound. For 90 generated dependency
DAGs, it checks that longest-path layering reproduces one sequential
topological reference for a specific monotone update, while merging the two
deepest layers changes at least one cell. A finite random search over shorter
phase assignments is retained only as an experiment-specific control.
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "topological_staging_results.json"

def make_dag(ell, width, seed, extra_edge_p=0.35):
    rng=np.random.default_rng(seed); layers=[]; nid=0
    for L in range(ell+1):
        k=1 if L in (0,ell) else 1+int(rng.integers(0,width))
        layers.append(list(range(nid,nid+k))); nid+=k
    G=nx.DiGraph(); G.add_nodes_from(range(nid))
    spine=[ns[0] for ns in layers]
    for a,b in zip(spine,spine[1:]): G.add_edge(a,b)   # selected path has ell edges
    for L in range(ell):
        for u in layers[L]:
            for v in layers[L+1]:
                if (u,v) not in G.edges and rng.random()<extra_edge_p: G.add_edge(u,v)
    return G

def run_sequential(G):
    P={c:list(G.predecessors(c)) for c in G.nodes}; x={c:1.0 for c in G.nodes}
    for c in nx.topological_sort(G): x[c]=1.0+sum(x[p] for p in P[c])
    return x

def run_staged(G, phase):
    P={c:list(G.predecessors(c)) for c in G.nodes}; x={c:1.0 for c in G.nodes}
    for ph in range(max(phase.values())+1):
        snap=dict(x); buf={}
        for c in G.nodes:
            if phase[c]==ph: buf[c]=1.0+sum(snap[p] for p in P[c])
        x.update(buf)   # barrier commit
    return x

def longest_path_layering(G):
    ph={c:0 for c in G.nodes}
    for c in nx.topological_sort(G):
        for p in G.predecessors(c): ph[c]=max(ph[c], ph[p]+1)
    return ph

def merge_top_two_layers(ph):
    K=max(ph.values()); return {c:(K-1 if v==K else v) for c,v in ph.items()}

def any_short_layering_works(G, seq, ell, tries=150, seed=0):
    rng=np.random.default_rng(seed); P={c:list(G.predecessors(c)) for c in G.nodes}
    best=0.0
    for _ in range(tries):
        ph={}
        for c in nx.topological_sort(G):
            lb=max([ph[p]+1 for p in P[c]] or [0])
            ph[c]=min(lb, ell-1) if lb>0 else int(rng.integers(0,ell))
        if max(ph.values())>ell-1: continue
        r=run_staged(G,ph); f=np.mean([abs(r[c]-seq[c])<1e-9 for c in G.nodes])
        best=max(best,f)
        if f==1.0: return True,1.0
    return False,best

if __name__=="__main__":
    rows=[]
    for ell in [2,3,4,5,6]:
        for width in [1,2,3]:
            for seed in range(6):
                G=make_dag(ell,width,seed*100+ell*7+width)
                if nx.dag_longest_path_length(G)!=ell: continue
                seq=run_sequential(G)
                ph_full=longest_path_layering(G); ph_short=merge_top_two_layers(ph_full)
                f_full=np.mean([abs(run_staged(G,ph_full)[c]-seq[c])<1e-9 for c in G.nodes])
                f_short=np.mean([abs(run_staged(G,ph_short)[c]-seq[c])<1e-9 for c in G.nodes])
                w,b=any_short_layering_works(G,seq,ell,seed=seed+1)
                rows.append(dict(ell=ell,width=width,seed=seed,nodes=G.number_of_nodes(),
                                 edges=G.number_of_edges(),barriers_needed=ell,
                                 full_correct=f_full,short_correct=f_short,
                                 any_short_works=w,best_short_frac=b))
    payload = {
        "scope": (
            "Generated-DAG check for one sequential-topological reference; "
            "not a lower-bound proof and not global snapshot semantics."
        ),
        "graphs": len(rows),
        "longest_path_layering_exact_all": all(r['full_correct']==1.0 for r in rows),
        "merged_deepest_layers_differs_all": all(r['short_correct']<1.0 for r in rows),
        "short_random_search_found_match": any(r['any_short_works'] for r in rows),
        "rows": rows,
    }
    print("graphs:",len(rows))
    print("longest-path layering exact:", payload["longest_path_layering_exact_all"])
    print("merged deepest layers differs:", payload["merged_deepest_layers_differs_all"])
    print("short random search found match:", payload["short_random_search_found_match"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")

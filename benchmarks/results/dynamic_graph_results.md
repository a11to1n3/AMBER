# Dynamic Graph Benchmark Runner Results

Synchronous bounded-confidence opinion dynamics with a deterministic sparse edge relation regenerated every step. Times are median wall-clock milliseconds; every non-skipped row is checked against the NumPy reference trajectory.

## Summary

- Status: `dynamic_graph_benchmark_runner_available`
- Rows: `63`
- Checked rows: `63`
- Skipped rows: `0`
- Reference mismatches: `0`
- Rows with raw n >= 5: `63`
- Final-like rows: `63` of `63`
- Frameworks: `AMBER, AgentPy, Agents.jl, Mesa, NumPy, Polars, Python object loop`

| seed | framework | mode | agents | raw n | median ms | IQR ms | final std | active edges | max abs diff vs NumPy | match |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | NumPy | dynamic_numpy_reference | 500 | 5 | 76.7 | 1.0 | 0.2204 | 2092 | 0.00e+00 | true |
| 42 | Polars | dynamic_columnar_polars | 500 | 5 | 95.5 | 8.1 | 0.2204 | 2092 | 0.00e+00 | true |
| 42 | Python object loop | dynamic_object_loop | 500 | 5 | 89.1 | 3.0 | 0.2204 | 2092 | 0.00e+00 | true |
| 42 | AMBER | dynamic_amber_object_container | 500 | 5 | 161.1 | 11.5 | 0.2204 | 2092 | 0.00e+00 | true |
| 42 | Mesa | dynamic_mesa_object_container | 500 | 5 | 97.9 | 6.8 | 0.2204 | 2092 | 0.00e+00 | true |
| 42 | AgentPy | dynamic_agentpy_object_container | 500 | 5 | 92.6 | 6.1 | 0.2204 | 2092 | 0.00e+00 | true |
| 42 | Agents.jl | dynamic_agentsjl_object_container | 500 | 5 | 2.8 | 0.1 | 0.2204 | 2092 | 0.00e+00 | true |
| 42 | NumPy | dynamic_numpy_reference | 1000 | 5 | 149.6 | 7.5 | 0.2158 | 4294 | 0.00e+00 | true |
| 42 | Polars | dynamic_columnar_polars | 1000 | 5 | 170.8 | 12.4 | 0.2158 | 4294 | 0.00e+00 | true |
| 42 | Python object loop | dynamic_object_loop | 1000 | 5 | 193.6 | 33.9 | 0.2158 | 4294 | 0.00e+00 | true |
| 42 | AMBER | dynamic_amber_object_container | 1000 | 5 | 362.1 | 25.4 | 0.2158 | 4294 | 0.00e+00 | true |
| 42 | Mesa | dynamic_mesa_object_container | 1000 | 5 | 186.5 | 5.9 | 0.2158 | 4294 | 0.00e+00 | true |
| 42 | AgentPy | dynamic_agentpy_object_container | 1000 | 5 | 194.6 | 39.3 | 0.2158 | 4294 | 0.00e+00 | true |
| 42 | Agents.jl | dynamic_agentsjl_object_container | 1000 | 5 | 6.4 | 0.4 | 0.2158 | 4294 | 0.00e+00 | true |
| 42 | NumPy | dynamic_numpy_reference | 5000 | 5 | 871.5 | 24.4 | 0.1847 | 20051 | 0.00e+00 | true |
| 42 | Polars | dynamic_columnar_polars | 5000 | 5 | 870.1 | 27.2 | 0.1847 | 20051 | 0.00e+00 | true |
| 42 | Python object loop | dynamic_object_loop | 5000 | 5 | 987.9 | 22.6 | 0.1847 | 20051 | 0.00e+00 | true |
| 42 | AMBER | dynamic_amber_object_container | 5000 | 5 | 3290.7 | 105.4 | 0.1847 | 20051 | 0.00e+00 | true |
| 42 | Mesa | dynamic_mesa_object_container | 5000 | 5 | 965.8 | 55.5 | 0.1847 | 20051 | 0.00e+00 | true |
| 42 | AgentPy | dynamic_agentpy_object_container | 5000 | 5 | 952.5 | 16.0 | 0.1847 | 20051 | 0.00e+00 | true |
| 42 | Agents.jl | dynamic_agentsjl_object_container | 5000 | 5 | 32.4 | 1.1 | 0.1847 | 20051 | 0.00e+00 | true |
| 77 | NumPy | dynamic_numpy_reference | 500 | 5 | 78.0 | 3.5 | 0.1959 | 2111 | 0.00e+00 | true |
| 77 | Polars | dynamic_columnar_polars | 500 | 5 | 88.4 | 1.1 | 0.1959 | 2111 | 0.00e+00 | true |
| 77 | Python object loop | dynamic_object_loop | 500 | 5 | 85.9 | 3.3 | 0.1959 | 2111 | 0.00e+00 | true |
| 77 | AMBER | dynamic_amber_object_container | 500 | 5 | 154.7 | 4.0 | 0.1959 | 2111 | 0.00e+00 | true |
| 77 | Mesa | dynamic_mesa_object_container | 500 | 5 | 86.2 | 12.1 | 0.1959 | 2111 | 0.00e+00 | true |
| 77 | AgentPy | dynamic_agentpy_object_container | 500 | 5 | 87.3 | 4.1 | 0.1959 | 2111 | 0.00e+00 | true |
| 77 | Agents.jl | dynamic_agentsjl_object_container | 500 | 5 | 2.9 | 0.1 | 0.1959 | 2111 | 0.00e+00 | true |
| 77 | NumPy | dynamic_numpy_reference | 1000 | 5 | 138.8 | 1.9 | 0.2303 | 3929 | 0.00e+00 | true |
| 77 | Polars | dynamic_columnar_polars | 1000 | 5 | 167.9 | 4.6 | 0.2303 | 3929 | 0.00e+00 | true |
| 77 | Python object loop | dynamic_object_loop | 1000 | 5 | 164.0 | 41.2 | 0.2303 | 3929 | 0.00e+00 | true |
| 77 | AMBER | dynamic_amber_object_container | 1000 | 5 | 321.3 | 12.5 | 0.2303 | 3929 | 0.00e+00 | true |
| 77 | Mesa | dynamic_mesa_object_container | 1000 | 5 | 169.2 | 0.4 | 0.2303 | 3929 | 0.00e+00 | true |
| 77 | AgentPy | dynamic_agentpy_object_container | 1000 | 5 | 168.3 | 0.3 | 0.2303 | 3929 | 0.00e+00 | true |
| 77 | Agents.jl | dynamic_agentsjl_object_container | 1000 | 5 | 6.1 | 0.3 | 0.2303 | 3929 | 0.00e+00 | true |
| 77 | NumPy | dynamic_numpy_reference | 5000 | 5 | 754.9 | 54.6 | 0.2281 | 19601 | 0.00e+00 | true |
| 77 | Polars | dynamic_columnar_polars | 5000 | 5 | 767.0 | 80.2 | 0.2281 | 19601 | 0.00e+00 | true |
| 77 | Python object loop | dynamic_object_loop | 5000 | 5 | 925.3 | 14.0 | 0.2281 | 19601 | 0.00e+00 | true |
| 77 | AMBER | dynamic_amber_object_container | 5000 | 5 | 3090.7 | 62.2 | 0.2281 | 19601 | 0.00e+00 | true |
| 77 | Mesa | dynamic_mesa_object_container | 5000 | 5 | 920.0 | 30.0 | 0.2281 | 19601 | 0.00e+00 | true |
| 77 | AgentPy | dynamic_agentpy_object_container | 5000 | 5 | 922.7 | 13.4 | 0.2281 | 19601 | 0.00e+00 | true |
| 77 | Agents.jl | dynamic_agentsjl_object_container | 5000 | 5 | 29.4 | 1.5 | 0.2281 | 19601 | 0.00e+00 | true |
| 123 | NumPy | dynamic_numpy_reference | 500 | 5 | 74.2 | 3.1 | 0.2062 | 2101 | 0.00e+00 | true |
| 123 | Polars | dynamic_columnar_polars | 500 | 5 | 92.4 | 5.4 | 0.2062 | 2101 | 0.00e+00 | true |
| 123 | Python object loop | dynamic_object_loop | 500 | 5 | 86.6 | 1.7 | 0.2062 | 2101 | 0.00e+00 | true |
| 123 | AMBER | dynamic_amber_object_container | 500 | 5 | 165.6 | 5.1 | 0.2062 | 2101 | 0.00e+00 | true |
| 123 | Mesa | dynamic_mesa_object_container | 500 | 5 | 93.3 | 8.2 | 0.2062 | 2101 | 0.00e+00 | true |
| 123 | AgentPy | dynamic_agentpy_object_container | 500 | 5 | 91.2 | 1.6 | 0.2062 | 2101 | 0.00e+00 | true |
| 123 | Agents.jl | dynamic_agentsjl_object_container | 500 | 5 | 2.9 | 0.1 | 0.2062 | 2101 | 0.00e+00 | true |
| 123 | NumPy | dynamic_numpy_reference | 1000 | 5 | 140.3 | 3.9 | 0.2205 | 3941 | 0.00e+00 | true |
| 123 | Polars | dynamic_columnar_polars | 1000 | 5 | 164.3 | 3.2 | 0.2205 | 3941 | 0.00e+00 | true |
| 123 | Python object loop | dynamic_object_loop | 1000 | 5 | 178.3 | 42.7 | 0.2205 | 3941 | 0.00e+00 | true |
| 123 | AMBER | dynamic_amber_object_container | 1000 | 5 | 325.4 | 8.5 | 0.2205 | 3941 | 0.00e+00 | true |
| 123 | Mesa | dynamic_mesa_object_container | 1000 | 5 | 167.8 | 1.8 | 0.2205 | 3941 | 0.00e+00 | true |
| 123 | AgentPy | dynamic_agentpy_object_container | 1000 | 5 | 170.4 | 2.7 | 0.2205 | 3941 | 0.00e+00 | true |
| 123 | Agents.jl | dynamic_agentsjl_object_container | 1000 | 5 | 5.6 | 0.3 | 0.2205 | 3941 | 0.00e+00 | true |
| 123 | NumPy | dynamic_numpy_reference | 5000 | 5 | 731.5 | 21.8 | 0.2144 | 19917 | 0.00e+00 | true |
| 123 | Polars | dynamic_columnar_polars | 5000 | 5 | 780.3 | 41.3 | 0.2144 | 19917 | 0.00e+00 | true |
| 123 | Python object loop | dynamic_object_loop | 5000 | 5 | 895.9 | 7.3 | 0.2144 | 19917 | 0.00e+00 | true |
| 123 | AMBER | dynamic_amber_object_container | 5000 | 5 | 3117.9 | 111.4 | 0.2144 | 19917 | 0.00e+00 | true |
| 123 | Mesa | dynamic_mesa_object_container | 5000 | 5 | 927.0 | 17.9 | 0.2144 | 19917 | 0.00e+00 | true |
| 123 | AgentPy | dynamic_agentpy_object_container | 5000 | 5 | 919.4 | 15.2 | 0.2144 | 19917 | 0.00e+00 | true |
| 123 | Agents.jl | dynamic_agentsjl_object_container | 5000 | 5 | 30.6 | 1.8 | 0.2144 | 19917 | 0.00e+00 | true |

## Interpretation

- The runner separates the dynamic graph workload from paper-only audits and records reproducible root benchmark rows.
- A zero-mismatch result supports semantic equivalence for synchronous step-varying graph coordination under this protocol.
- This remains one-machine timing evidence until platform replication is added.

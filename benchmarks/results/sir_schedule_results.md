# SIR Schedule Benchmark Runner

Deterministic split sync/async SIR rows emitted from the root benchmark runner.

## Summary

- Status: `sir_schedule_benchmark_runner_available`
- Rows: `45`
- Checked rows: `42`
- Skipped rows: `3`
- Budget-skipped rows: `3`
- Reference mismatches: `0`
- Rows with raw n >= 10: `42`
- Headline-size 10-run rows: `42`
- Headline-size 10-run rows by framework: `{'AMBER (loop)': 6, 'AMBER (vectorized)': 3, 'AgentPy': 6, 'Agents.jl': 6, 'Melodie': 6, 'Mesa': 6, 'NumPy': 3, 'SimPy': 3, 'Spatial reference': 3}`

| mode | framework | schedule | steps | agents | raw n | median ms | final S/I/R | reference match |
|---|---|---|---:|---:|---:|---:|---|---|
| async_spatial_reference | Spatial reference | async | 50 | 500 | 10 | 62.5 | 154/89/257 | True |
| agentsjl_actual_source_async | Agents.jl | async | 50 | 500 | 10 | 2.3 | 154/89/257 | True |
| async_amber_loop_actual | AMBER (loop) | async | 50 | 500 | 10 | 124.8 | 154/89/257 | True |
| async_mesa_actual | Mesa | async | 50 | 500 | 10 | 112.5 | 154/89/257 | True |
| async_agentpy_actual | AgentPy | async | 50 | 500 | 10 | 111.6 | 154/89/257 | True |
| async_melodie_actual | Melodie | async | 50 | 500 | 10 | 112.7 | 154/89/257 | True |
| async_simpy_refactored | SimPy | async | 50 | 500 | 0 | skipped | - | Excluded by declared run-budget policy: asynchronous refactored-SimPy is event-scheduled and remains outside the measured final grid unless a larger timeout budget is allocated; all other split-SIR headline rows are measured with 10 raw samples. |
| sync_numpy_reference | NumPy | sync | 50 | 500 | 10 | 28.0 | 195/82/223 | True |
| sync_amber_vectorized_view | AMBER (vectorized) | sync | 50 | 500 | 10 | 47.9 | 195/82/223 | True |
| agentsjl_actual_source_sync | Agents.jl | sync | 50 | 500 | 10 | 1.6 | 195/82/223 | True |
| sync_amber_loop_actual | AMBER (loop) | sync | 50 | 500 | 10 | 366.4 | 195/82/223 | True |
| sync_mesa_actual | Mesa | sync | 50 | 500 | 10 | 346.7 | 195/82/223 | True |
| sync_agentpy_actual | AgentPy | sync | 50 | 500 | 10 | 349.0 | 195/82/223 | True |
| sync_melodie_actual | Melodie | sync | 50 | 500 | 10 | 341.4 | 195/82/223 | True |
| sync_simpy_refactored | SimPy | sync | 50 | 500 | 10 | 324.7 | 195/82/223 | True |
| async_spatial_reference | Spatial reference | async | 50 | 1000 | 10 | 153.6 | 3/7/990 | True |
| agentsjl_actual_source_async | Agents.jl | async | 50 | 1000 | 10 | 14.5 | 3/7/990 | True |
| async_amber_loop_actual | AMBER (loop) | async | 50 | 1000 | 10 | 471.9 | 3/7/990 | True |
| async_mesa_actual | Mesa | async | 50 | 1000 | 10 | 430.5 | 3/7/990 | True |
| async_agentpy_actual | AgentPy | async | 50 | 1000 | 10 | 434.4 | 3/7/990 | True |
| async_melodie_actual | Melodie | async | 50 | 1000 | 10 | 431.2 | 3/7/990 | True |
| async_simpy_refactored | SimPy | async | 50 | 1000 | 0 | skipped | - | Excluded by declared run-budget policy: asynchronous refactored-SimPy is event-scheduled and remains outside the measured final grid unless a larger timeout budget is allocated; all other split-SIR headline rows are measured with 10 raw samples. |
| sync_numpy_reference | NumPy | sync | 50 | 1000 | 10 | 76.9 | 10/23/967 | True |
| sync_amber_vectorized_view | AMBER (vectorized) | sync | 50 | 1000 | 10 | 49.8 | 10/23/967 | True |
| agentsjl_actual_source_sync | Agents.jl | sync | 50 | 1000 | 10 | 7.7 | 10/23/967 | True |
| sync_amber_loop_actual | AMBER (loop) | sync | 50 | 1000 | 10 | 1255.2 | 10/23/967 | True |
| sync_mesa_actual | Mesa | sync | 50 | 1000 | 10 | 1144.5 | 10/23/967 | True |
| sync_agentpy_actual | AgentPy | sync | 50 | 1000 | 10 | 1187.8 | 10/23/967 | True |
| sync_melodie_actual | Melodie | sync | 50 | 1000 | 10 | 1147.4 | 10/23/967 | True |
| sync_simpy_refactored | SimPy | sync | 50 | 1000 | 10 | 1140.2 | 10/23/967 | True |
| async_spatial_reference | Spatial reference | async | 50 | 5000 | 10 | 996.8 | 0/0/5000 | True |
| agentsjl_actual_source_async | Agents.jl | async | 50 | 5000 | 10 | 311.7 | 0/0/5000 | True |
| async_amber_loop_actual | AMBER (loop) | async | 50 | 5000 | 10 | 7124.7 | 0/0/5000 | True |
| async_mesa_actual | Mesa | async | 50 | 5000 | 10 | 7114.3 | 0/0/5000 | True |
| async_agentpy_actual | AgentPy | async | 50 | 5000 | 10 | 6527.0 | 0/0/5000 | True |
| async_melodie_actual | Melodie | async | 50 | 5000 | 10 | 6555.2 | 0/0/5000 | True |
| async_simpy_refactored | SimPy | async | 50 | 5000 | 0 | skipped | - | Excluded by declared run-budget policy: asynchronous refactored-SimPy is event-scheduled and remains outside the measured final grid unless a larger timeout budget is allocated; all other split-SIR headline rows are measured with 10 raw samples. |
| sync_numpy_reference | NumPy | sync | 50 | 5000 | 10 | 233.0 | 0/0/5000 | True |
| sync_amber_vectorized_view | AMBER (vectorized) | sync | 50 | 5000 | 10 | 428.6 | 0/0/5000 | True |
| agentsjl_actual_source_sync | Agents.jl | sync | 50 | 5000 | 10 | 156.4 | 0/0/5000 | True |
| sync_amber_loop_actual | AMBER (loop) | sync | 50 | 5000 | 10 | 13171.4 | 0/0/5000 | True |
| sync_mesa_actual | Mesa | sync | 50 | 5000 | 10 | 13693.5 | 0/0/5000 | True |
| sync_agentpy_actual | AgentPy | sync | 50 | 5000 | 10 | 13255.6 | 0/0/5000 | True |
| sync_melodie_actual | Melodie | sync | 50 | 5000 | 10 | 13256.7 | 0/0/5000 | True |
| sync_simpy_refactored | SimPy | sync | 50 | 5000 | 10 | 13353.1 | 0/0/5000 | True |

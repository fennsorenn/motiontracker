"""Batched torch (GPU) version of the M1 training-data generator.

The numpy generator (`mttrain.dataset.Generator` with `seated`, `synth`, `noise`, `mpstyle`,
`camera`, `features`, `skeleton`) is the reference; these modules mirror it on whole batches so
training needs no CPU data workers (2026-11-08: CPU generation needed ~10 processes and, with
pymomentum's own thread pools, overloaded the machine). Deterministic parts are tested to match the
reference exactly; random parts are tested to match it statistically.
"""

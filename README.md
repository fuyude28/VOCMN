# VOCMN minimal reproducibility package

This is a small, runnable release of the variable-order concise memory network
(VOCMN) implementation used for the manuscript's synthetic validation and the
optional maritime input workflow. It contains source code and the maritime
trajectory input, but no generated figures, logs, notebooks, or previous result
tables.

## Contents

```text
tools/                         Core VOCMN, CMN, generator, and factorisation code
variable_order_experiments.py  Oracle-leaf synthetic compression experiment
end_to_end_vocmn_experiments.py
                               Train/validation/test synthetic experiment
vocmn_support_pilot.py         Synthetic support-coverage diagnostic
ship_community_experiment.py   Optional maritime graph construction
data/ship_VOCMN.csv            Maritime trajectory input
test_*.py                      Focused regression tests
```

## Requirements

Python 3.9 or newer is required. The tested environment used the versions in
`requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

## Synthetic smoke run

The following command completes quickly on a laptop and writes only derived
files under `results/synthetic_smoke/`:

```bash
python variable_order_experiments.py \
  --n-nodes 6 --n-modes 2 --n-paths 4 --path-length 30 \
  --conditions mixed --orders 3 --dirichlet-concentrations 1.0 \
  --repeats 1 --max-rank 2 --n-candidates 1 --n-null 2 \
  --out-dir results/synthetic_smoke
```

Expected outputs are `fig2_main.png`, `metrics.csv`, and, when a parquet
engine is available, `metrics.parquet`.

For the manuscript-scale finite-sampling experiment, use the same script with
the paper's settings (`--n-nodes 10 --n-paths 200 --path-length 300
--repeats 25 --n-modes 2 5 10 --dirichlet-concentrations 0.5 1.0 1.5`).
That run is substantially more expensive and is intentionally not executed by
the smoke command.

The train-only VOCMN/CMN comparison can be smoke-tested with:

```bash
python end_to_end_vocmn_experiments.py \
  --n-nodes 6 --n-paths 9 --path-length 40 \
  --conditions first second third mixed --n-modes 2 \
  --dirichlet-concentrations 1.0 --repeats 1 \
  --max-rank 1 --n-candidates 1 \
  --out-dir results/end_to_end_smoke
```

## Tests

```bash
python -m unittest test_vocmn_model.py test_variable_order_dynamics.py
```

The tests cover count construction, suffix regularisation, order selection,
prediction, state-network export, and deterministic synthetic generation.

## Optional maritime workflow

No maritime result files are included. To construct the first-order,
uncompressed variable-order, and compressed VOCMN graphs from the included
input:

```bash
python ship_community_experiment.py \
  --input data/ship_VOCMN.csv \
  --out-dir results/maritime_graphs
```

This is a data-processing workflow, not a reproduction of every manuscript
figure. The input provenance and redistribution caveat are recorded in
`data/README.md`.

## Reproducibility notes

Synthetic paths and experiment grids use explicit NumPy seeds. All paths and
outputs are relative to the repository, so the commands above work from a
clean clone. The full maritime analyses may require substantially more memory
and runtime than the smoke runs.

## Citation and license

Please cite the associated manuscript and retain the MIT license in `LICENSE`.
The core implementation includes the original 2025 attribution from the
upstream project.

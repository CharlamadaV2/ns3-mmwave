# ARPO placement planner source — provenance

Origin: the geometric and optimization placement planners of the ARPO network
optimization model, supplied to this project as a local source bundle.

Date copied: 2026-09-29.

Authorization: on 2026-09-29 the user authorized publishing these nine
algorithm source files in this repository. The authorization covers only the
files listed below. It does not cover the supplied RF configuration
(`rf_config.yaml`), datasets, examples, tests, weights, generated results,
archives, or any other file from the supplied bundle, none of which are copied
here. No license designation was supplied, and none is claimed.

The files are byte-for-byte copies. No imports, algorithms, or other content
were edited. They are kept as an unused reference copy for audits; nothing in
mesh-sim imports or runs them (see Adaptation below).

| File | SHA-256 |
| --- | --- |
| `models.py` | `cc5a667fba9fba674e3d1cf3c99be227dcfeb0030ad32673a3ea9b85cf0c717c` |
| `core/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `core/geometry.py` | `89a76713e0f8abe1543ebe65d4bbe505596c5319a914be521a99beebba3cef73` |
| `core/rf.py` | `ef192e3757aa0a75c59c288dbe6461f027b9678da62ac43a19c0a72827a77178` |
| `core/context.py` | `c772c70d814b4b2564e70d38bf8874995443603faa32d0215b3e936c3b803dfe` |
| `planners/__init__.py` | `9f5747675581b5d6727a92597e87989059b3f7cf4b11e39652e5592a58e11304` |
| `planners/base.py` | `99c063a3a9ba12cae71eba6140061e46825cd71773665944037ddb62057166b6` |
| `planners/geometric.py` | `ee757cc5ce6e4cdf94d86149f335bcf07214f89f738965de3b85f8a96a280966` |
| `planners/optimization.py` | `24ef728dfeaf77aacd3d3c62d4f5381634d07c850c1eaf2be1171515ffc1db8c` |

## Adaptation

The committed reference files above are unchanged and unused at runtime;
`scripts/baselines/tests/test_provenance.py` checks the hashes in this table and
that no adapted module refers to this directory. The placement baselines run
adapted ports that depend only on `numpy` (pinned by `requirements.txt`), with
no pydantic, shapely, pyproj, PyYAML, lat/lon frame or RF YAML. Their identity
is the aggregate `planner_code_sha256` in `baseline_manifest.json`, not these
hashes. The adapted algorithms are described in
[`scripts/baselines/planners/README.md`](../../scripts/baselines/planners/README.md).

| Reference | Adapted into |
| --- | --- |
| `core/rf.py` `MovementCost` (`cost`, `within_cap`) | `scripts/baselines/planners/objective.py` `MovementCost` |
| `core/geometry.py` `grid_points_in` (rectangle case) | `objective.py` `rectangle_grid` |
| `core/context.py` `adjacency`, `connected_to_gateway`, `coverage_mask`, `survives_single_loss`, `controlled_mesh_survives_single_loss`; `planners/base.py` `_diagnose`; `planners/optimization.py` `_Evaluator.score` | `objective.py` components/core, coverage union, vulnerability pairs, `score`, `diagnose` |
| `planners/geometric.py` `GeometricModel` (`_order_movable`, `_candidates`, `_greedy_place`, `_feasible`, `_connected_set`) | `scripts/baselines/planners/geometric.py` |
| `planners/optimization.py` `OptimizationModel` (`solve`, `_propose`, `_clamp`) and the `core/rf.py` `OptimizerConfig` constants | `scripts/baselines/planners/optimization.py` |

Deviations from the reference:

- Gateway-free: no C2, gateway, fixed-anchor or relay role. The connected core
  is the largest connected component of all mesh nodes (ties: smallest roster
  index), and greedy anchors are unselected peers in that core plus previously
  placed selected peers, recomputed after each choice; an empty pool allows a
  zero-anchor bootstrap, including when every node is movable.
- Links and coverage are scored by the simulator channel (`--channel-query`,
  `scripts/baselines/planners/channel.py`) instead of the Friis `RFEngine`.
  Coverage is the clipped-cell area of probe receivers on a grid covered by any
  core node, not a ground-footprint radius.
- Resilience counts vulnerability pairs: for each selected core victim, the
  unordered core pairs it disconnects. This replaces the (victim, orphan)
  count rooted at the gateway.
- The optimizer's `swap` move exchanges x/y only (each node keeps its z), and
  the `altitude` move is removed; nudge/teleport/swap weights are renormalized
  (0.55/0.20/0.10 divided by 0.85). Placement is 2-D at each node's own z.
- Selected nodes are processed in roster order, not the reference's
  movable-C2-first, longest-radio-range-first order.
- The RF YAML and its optimizer section are retired: termination is
  `[baseline] max_iterations` only (no time budget), and the seed is
  `[baseline] seed`.
- Movement penalties and caps come from the `[baseline]` `*_fixed_cost_m2`,
  `*_cost_m2_per_m` and `*_max_displacement_m` keys (reference defaults
  150000 / 500 aerial / 100 ground, no cap), not from `movement_cost` in the RF
  file.

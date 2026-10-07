@page scripts_baselines_planners scripts/baselines/planners

@brief The geometric and optimization placement planners and their shared objective and channel scorer.

The two placement baselines, `geometric` (sequential greedy) and
`optimization` (simulated annealing), adapted from the Desktop reference copy
in `third_party/arpo_placement/` (read-only; nothing imports it). Every
candidate layout is scored by the simulator's own channel through the
`--channel-query` worker, so there is no Friis model and no RF YAML here.

| File | Owns |
| --- | --- |
| `channel.py` | `ChannelScorer`: the only client of the `mesh_channel_query_v1` worker (launch, init check, batching, timeouts, cleanup, stats). |
| `objective.py` | Grids, components/core, probe coverage, vulnerability pairs, `MovementCost`, the shared `score`, `diagnose`, candidate positions. Pure numpy. |
| `geometric.py` | Greedy peer-anchor placement. |
| `optimization.py` | Keep-best annealing, warm-started from `geometric`. |

Both strategies take `(request, scorer, log)` with the scorer's probes already
set by `objective.prepare_scorer`, and return `(positions N x 3, diagnostics,
notes)`. Each module's `settings(request)` lists the constants used, for the
manifest's `planner_settings`.

## Shared objective

All terms are m² so the Desktop penalty values keep their units. Each layout
is one full query: the t = 0 mesh link table plus every node-to-probe link.
`connected` comes from the worker; Python never re-derives a threshold.

- **Core.** Connected components over all mesh nodes; the core is the largest,
  ties going to the component holding the smallest roster index. No root.
- **Coverage.** Probes sit at the clipped-cell centres of the coverage grid and
  carry their clipped areas (the areas sum to the AOI). A probe counts once if
  any core node covers it. Nodes outside the core contribute nothing.
- **Connectivity.** `BIG = 2 x AOI` per selected node outside the core.
  Unselected nodes are never charged.
- **Vulnerability pairs.** For each victim in core ∩ selected, remove it,
  recompute components and count unordered core pairs (victim excluded) that
  end up apart. Weight per pair: `resilience` AOI, `balanced` 0.02 x AOI,
  `coverage` 0.
- **Separation tie-break.** `0.4 x cell_m² x mean over selected of
  min(nearest-neighbour distance / rectangle diagonal, 1)`; always below one
  coverage cell.
- **Movement.** Per selected node, from its ORIGINAL start:
  `0` if displacement ≤ 1e-6 m, else `fixed_cost_m2 + cost_m2_per_m x d`.
  `max_displacement_m` is a hard cap on candidates and proposals.

`total = coverage + separation - movement - BIG x disconnected - w x vuln`.
`diagnose` adds component sizes, core ids, `survives_single_node_loss`
(victims = every core member), `controlled_mesh_survives_single_loss`
(victims = core ∩ selected), per-node displacement and cost, and
`fixed_cost_m2 / AOI` per platform. A ratio ≥ 1 is logged as a scale warning,
never an error.

Grid cell size follows the Desktop rule `max(min_resolution_m,
sqrt(AOI / cells))`, with cells laid from the rectangle's lower-left corner.
Unlike the Desktop grid, every cell that overlaps the rectangle is kept and
the last row/column is clipped, so edge cells are weighted by their real area.
The candidate grid uses the same rule with `candidate_cells`.

## Geometric: sequential greedy

1. Query the start layout and score it.
2. Visit selected nodes once each, in roster order. For node `i`, the anchor
   pool is the current core's unselected members plus already-visited
   selected members (at their chosen position, stay-put included), minus `i`.
   The pool is recomputed from the current core before every node, so it is
   always one component.
3. Needs: `coverage` 1 per node, `resilience` 2, `balanced`
   `ceil(n x balanced_core_fraction)` twos then ones.
   `eff_need = min(need, pool size)`; an empty pool is a zero-anchor
   bootstrap (an all-movable roster starts this way).
4. Candidates are candidate-grid points inside the rectangle, the evaluation
   `[rl]` bounds, the node's random-walk bounds and its cap. All candidate
   layouts for the node go out in one batched query. A candidate is feasible
   when the queried `connected` matrix links `i` to at least `eff_need` pool
   members; with none at 2, the need drops to 1.
5. Each feasible layout is scored in full. The best one is taken only if its
   total beats the current layout's by more than 1e-6 m²; otherwise the node
   stays put. Stay-put is always allowed, even when no candidate is feasible.

Anchor links only filter candidates. Whether the result survives a node loss
is decided by the vulnerability term and `diagnose`, not by link counts.
Results are cached by exact layout bytes; there are no partial or
changed-pairs-only queries, because sampled draws depend on evaluation order
(see [src/query](@ref src_query)).

## Optimization: keep-best annealing

1. Run `geometric` with the same objective, then query the greedy layout and
   the start layout (two queries) and continue from the better one (ties go
   to the greedy layout). The best layout seen is returned.
2. Each iteration draws a move with weights renormalised after dropping the
   Desktop `altitude` move: nudge 0.55/0.85, teleport 0.20/0.85, swap
   0.10/0.85.
   - `nudge`: Gaussian x/y step on one selected node.
   - `teleport`: one selected node to a random uncovered probe plus a Gaussian
     jitter of 0.15 x sigma.
   - `swap`: two selected nodes exchange x/y only; each keeps its own z.
     With one selected node it acts as a nudge.
3. Every moved node is pulled back inside its cap (to 0.999 x cap, measured
   from its original start), then the proposal is rejected without a query
   if any moved node is outside the rectangle, the evaluation `[rl]` bounds or
   its random-walk bounds.
4. Each remaining proposal is one full query. Metropolis acceptance at
   `T = T0 (Tf / T0)^frac`, with `T0 = 0.02 x AOI`, `Tf = 0.0002 x AOI`,
   `sigma = 0.08 x diagonal x 0.05^frac + 2 m`, and `frac = iteration /
   max_iterations`. The run stops after `max_iterations`; there is no time
   budget. The RNG is `numpy.random.default_rng(planner seed)`.

For a fixed worker and seed, a run is repeatable on one machine. Different
layouts can consume different numbers of channel draws, so the objective is
not a smooth function of position.

## Differences from the Desktop reference

| Desktop | Here |
| --- | --- |
| C2/gateway root; reachability by BFS from it | Rootless largest-component core; no gateway, fixed anchor or BLOS terminal. |
| Order: movable C2 first, then longest radio range | Roster order. |
| Anchors: C2 plus satellite terminals plus placed movables; connected relays as 1-anchors | Pool from the current core only (unselected members plus visited selected members); zero-anchor bootstrap when empty. |
| Disc coverage from Friis ranges on uniform cells | Probe receivers scored by the simulator channel; clipped-cell areas. |
| Greedy gain: new coverage minus move cost plus separation of the node alone | Full-layout score delta, including connectivity, vulnerability and every selected node's movement cost. |
| Pure-relay nodes push the frontier; soft costs only for coverage radios | Every mesh node is a coverage node; soft costs apply to all selected nodes. |
| (victim, orphan) counts from the gateway; sole backhaul terminal skipped | (victim, split core pair) counts over the core. |
| Moves include `altitude`; `swap` exchanges positions and altitudes | No altitude move; `swap` is x/y only; z never changes. |
| Swap over the cap is rejected | Both swap participants are clamped like single-node moves. |
| Time budget `budget_s_per_coa` or `max_iters` | `max_iterations` only. |
| RF/optimizer YAML (`rf_config`) | Retired; constants live in `optimization.py`, penalties in `[baseline]`. |
| Altitude chosen per node from its AGL band | z fixed at each node's start. |

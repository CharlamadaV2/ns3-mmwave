"""mesh-sim plotting module.

Generates time-series and summary plots from mesh-sim CSV/JSON outputs,
with multi-seed aggregation and 95% confidence intervals.
"""

from .loaders import (
    discover_seed_dirs,
    load_flows_csv,
    load_links_csv,
    load_mcs_csv,
    load_positions_csv,
    load_routes_csv,
    load_rx_power_csv,
    load_summary,
)
from .aggregation import aggregate_summaries, aggregate_timeseries
from .plots_timeseries import (
    plot_capacity_timeseries,
    plot_derived_geometry,
    plot_latency_timeseries,
    plot_mcs_timeseries,
    plot_rx_power_timeseries,
    plot_sinr_timeseries,
    plot_throughput_timeseries,
)
from .plots_summary import (
    plot_network_summary,
    plot_per_flow_bars,
    plot_per_node_bars,
    plot_sim_runtime,
)

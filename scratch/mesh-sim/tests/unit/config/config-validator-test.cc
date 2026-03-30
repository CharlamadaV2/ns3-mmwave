/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Unit tests for config validation and seed parsing.
 * Standalone binary -- no ns-3 dependency.
 *
 * Build:  see mesh-sim-config-test target in CMakeLists.txt
 * Run:    ./mesh-sim-config-test
 */

#include "src/config/config-validator.h"
#include "src/util/string-utils.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace mesh_sim;

// ---- helpers ----

static int g_pass = 0;
static int g_fail = 0;

static void
check(bool cond, const std::string& name)
{
    if (cond)
    {
        ++g_pass;
    }
    else
    {
        ++g_fail;
        std::cerr << "FAIL: " << name << "\n";
    }
}

static SimConfig
makeValid()
{
    SimConfig cfg;
    cfg.scenario_name = "test";
    cfg.seed       = 42;
    cfg.run_id     = 1;
    cfg.duration_s = 10.0;
    cfg.warmup_s   = 0.0;
    cfg.tick_s     = 0.1;

    cfg.channel.frequency_ghz = 28.0;
    cfg.channel.bandwidth_mhz = 400.0;
    cfg.channel.channel_model  = "3gpp";
    cfg.channel.scenario       = "UMi";

    cfg.mesh.traffic.model         = "constant";
    cfg.mesh.traffic.demand_mbps   = 10.0;
    cfg.mesh.traffic.flow_topology = "all_pairs";
    cfg.mesh.routing.algorithm     = "shortest_path";

    NodeSpec n1;
    n1.id       = "node0";
    n1.mobility = "fixed";
    NodeSpec n2;
    n2.id       = "node1";
    n2.mobility = "fixed";
    cfg.nodes.push_back(n1);
    cfg.nodes.push_back(n2);

    return cfg;
}

static bool
hasError(const ValidationResult& r, const std::string& substr)
{
    for (const auto& e : r.errors)
    {
        if (e.find(substr) != std::string::npos)
            return true;
    }
    return false;
}

// ---- config validation tests ----

static void
test_valid_config()
{
    auto r = ValidateConfig(makeValid());
    check(r.ok(), "valid config should pass");
}

static void
test_zero_duration()
{
    auto cfg = makeValid();
    cfg.duration_s = 0.0;
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "zero duration rejected");
    check(hasError(r, "duration_s"), "error mentions duration_s");
}

static void
test_negative_duration()
{
    auto cfg = makeValid();
    cfg.duration_s = -1.0;
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "negative duration rejected");
}

static void
test_zero_tick()
{
    auto cfg = makeValid();
    cfg.tick_s = 0.0;
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "zero tick rejected");
}

static void
test_tick_exceeds_duration()
{
    auto cfg = makeValid();
    cfg.tick_s = 20.0;
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "tick > duration rejected");
    check(hasError(r, "tick_s"), "error mentions tick_s");
}

static void
test_negative_warmup()
{
    auto cfg = makeValid();
    cfg.warmup_s = -1.0;
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "negative warmup rejected");
}

static void
test_warmup_exceeds_duration()
{
    auto cfg = makeValid();
    cfg.warmup_s = 10.0;  // == duration_s
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "warmup >= duration rejected");
}

static void
test_too_few_nodes()
{
    auto cfg = makeValid();
    cfg.nodes.clear();
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "zero nodes rejected");
    check(hasError(r, "2 nodes"), "error mentions 2 nodes");

    cfg.nodes.push_back(NodeSpec{"solo", "peer", "fixed", {}, {}, {}});
    r = ValidateConfig(cfg);
    check(!r.ok(), "one node rejected");
}

static void
test_unknown_traffic_model()
{
    auto cfg = makeValid();
    cfg.mesh.traffic.model = "burst";
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "unknown traffic model rejected");
    check(hasError(r, "traffic.model"), "error mentions traffic.model");
}

static void
test_unknown_flow_topology()
{
    auto cfg = makeValid();
    cfg.mesh.traffic.flow_topology = "star";
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "unknown flow topology rejected");
}

static void
test_unknown_routing_algorithm()
{
    auto cfg = makeValid();
    cfg.mesh.routing.algorithm = "aodv";
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "unknown routing algorithm rejected");
}

static void
test_unknown_channel_scenario()
{
    auto cfg = makeValid();
    cfg.channel.scenario = "Rural";
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "unknown channel scenario rejected");
}

static void
test_unknown_channel_model()
{
    auto cfg = makeValid();
    cfg.channel.channel_model = "ray_trace";
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "unknown channel model rejected");
}

static void
test_unknown_mobility()
{
    auto cfg = makeValid();
    cfg.nodes[0].mobility = "teleport";
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "unknown mobility rejected");
    check(hasError(r, "node0"), "error mentions the node ID");
}

static void
test_negative_bandwidth()
{
    auto cfg = makeValid();
    cfg.channel.bandwidth_mhz = -10.0;
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "negative bandwidth rejected");
}

static void
test_negative_frequency()
{
    auto cfg = makeValid();
    cfg.channel.frequency_ghz = 0.0;
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "zero frequency rejected");
}

static void
test_negative_demand()
{
    auto cfg = makeValid();
    cfg.mesh.traffic.demand_mbps = -5.0;
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "negative demand rejected");
}

static void
test_gateway_missing_id()
{
    auto cfg = makeValid();
    cfg.mesh.traffic.flow_topology = "gateway";
    cfg.mesh.traffic.gateway_node_id = "";
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "gateway without ID rejected");
    check(hasError(r, "gateway_node_id"), "error mentions gateway_node_id");
}

static void
test_gateway_nonexistent_node()
{
    auto cfg = makeValid();
    cfg.mesh.traffic.flow_topology = "gateway";
    cfg.mesh.traffic.gateway_node_id = "nonexistent";
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "gateway with bad node ID rejected");
    check(hasError(r, "nonexistent"), "error mentions the bad ID");
}

static void
test_gateway_valid()
{
    auto cfg = makeValid();
    cfg.mesh.traffic.flow_topology = "gateway";
    cfg.mesh.traffic.gateway_node_id = "node0";
    auto r = ValidateConfig(cfg);
    check(r.ok(), "gateway with valid node ID accepted");
}

static void
test_random_pairs_zero_count()
{
    auto cfg = makeValid();
    cfg.mesh.traffic.flow_topology = "random_pairs";
    cfg.mesh.traffic.random_pair_count = 0;
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "random_pairs with zero count rejected");
}

static void
test_building_inverted_bounds()
{
    auto cfg = makeValid();
    BuildingSpec b;
    b.id    = "bad-building";
    b.x_min = 10.0;
    b.x_max = 5.0;  // inverted
    b.y_min = 0.0;
    b.y_max = 10.0;
    b.z_min = 0.0;
    b.z_max = 10.0;
    cfg.buildings.push_back(b);
    auto r = ValidateConfig(cfg);
    check(!r.ok(), "inverted building bounds rejected");
    check(hasError(r, "x_min"), "error mentions x_min");
}

static void
test_multiple_errors()
{
    auto cfg = makeValid();
    cfg.duration_s = 0.0;
    cfg.channel.frequency_ghz = -1.0;
    cfg.mesh.traffic.model = "invalid";
    auto r = ValidateConfig(cfg);
    check(r.errors.size() >= 3, "multiple errors reported at once");
}

// ---- seed parsing tests ----

static void
test_seed_single()
{
    auto seeds = parseSeedList("42");
    check(seeds.size() == 1 && seeds[0] == 42, "single seed");
}

static void
test_seed_multiple()
{
    auto seeds = parseSeedList("1,2,3");
    check(seeds.size() == 3, "multiple seeds count");
    check(seeds[0] == 1 && seeds[1] == 2 && seeds[2] == 3, "multiple seeds values");
}

static void
test_seed_empty()
{
    auto seeds = parseSeedList("");
    check(seeds.empty(), "empty string returns empty vector");
}

static void
test_seed_skip_empty_tokens()
{
    auto seeds = parseSeedList("1,,3");
    check(seeds.size() == 2, "empty tokens skipped count");
    check(seeds[0] == 1 && seeds[1] == 3, "empty tokens skipped values");
}

// ---- main ----

int
main()
{
    // Config validation
    test_valid_config();
    test_zero_duration();
    test_negative_duration();
    test_zero_tick();
    test_tick_exceeds_duration();
    test_negative_warmup();
    test_warmup_exceeds_duration();
    test_too_few_nodes();
    test_unknown_traffic_model();
    test_unknown_flow_topology();
    test_unknown_routing_algorithm();
    test_unknown_channel_scenario();
    test_unknown_channel_model();
    test_unknown_mobility();
    test_negative_bandwidth();
    test_negative_frequency();
    test_negative_demand();
    test_gateway_missing_id();
    test_gateway_nonexistent_node();
    test_gateway_valid();
    test_random_pairs_zero_count();
    test_building_inverted_bounds();
    test_multiple_errors();

    // Seed parsing
    test_seed_single();
    test_seed_multiple();
    test_seed_empty();
    test_seed_skip_empty_tokens();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.\n";
    if (g_fail > 0)
    {
        std::cout << "SOME TESTS FAILED.\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}

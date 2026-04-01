/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Unit tests for SinrToCapacity and LinkTable.
 * Standalone binary -- no ns-3 dependency.
 *
 * Build:  make -f Makefile.eval
 * Run:    ./eval-test
 */

#include "src/eval/sinr-capacity.h"
#include "src/eval/link-table.h"

#include <cassert>
#include <cmath>
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

static bool
approx(double a, double b, double tol = 0.01)
{
    return std::fabs(a - b) < tol;
}

// ---- SinrToCapacity tests ----

static void
test_below_min_sinr_returns_zero()
{
    double cap = SinrToCapacity(-10.0, 400e6, "shannon");
    check(cap == 0.0, "shannon below SINR_MIN returns 0");

    cap = SinrToCapacity(-10.0, 400e6, "table");
    check(cap == 0.0, "table below SINR_MIN returns 0");
}

static void
test_shannon_positive_sinr()
{
    // At 0 dB SINR (linear=1), Shannon: C = B * log2(2) = B * 1 = 400 Mbps
    double cap = SinrToCapacity(0.0, 400e6, "shannon");
    check(approx(cap, 400.0, 1.0), "shannon 0 dB SINR ~= 400 Mbps");
}

static void
test_shannon_high_sinr()
{
    // At 20 dB (linear=100), Shannon: C = 400e6 * log2(101) ~= 2664 Mbps
    double cap = SinrToCapacity(20.0, 400e6, "shannon");
    check(cap > 2600.0 && cap < 2700.0, "shannon 20 dB SINR in expected range");
}

static void
test_shannon_increases_with_sinr()
{
    double cap_low  = SinrToCapacity(0.0, 400e6, "shannon");
    double cap_high = SinrToCapacity(10.0, 400e6, "shannon");
    check(cap_high > cap_low, "shannon capacity increases with SINR");
}

static void
test_shannon_scales_with_bandwidth()
{
    double cap_200 = SinrToCapacity(10.0, 200e6, "shannon");
    double cap_400 = SinrToCapacity(10.0, 400e6, "shannon");
    check(approx(cap_400, 2.0 * cap_200, 0.1), "shannon capacity scales linearly with BW");
}

static void
test_table_lowest_mcs()
{
    // At exactly -6.7 dB, should get spectral_eff = 0.15
    double cap = SinrToCapacity(-6.7, 400e6, "table");
    check(approx(cap, 0.15 * 400, 0.1), "table at lowest MCS entry");
}

static void
test_table_mid_range()
{
    // At 10.3 dB, spectral_eff = 2.41 -> capacity = 2.41 * 400 = 964 Mbps
    double cap = SinrToCapacity(10.3, 400e6, "table");
    check(approx(cap, 2.41 * 400, 0.1), "table at mid-range MCS");
}

static void
test_table_highest_mcs()
{
    // At 30 dB (well above max), should clamp to highest entry: 5.55
    double cap = SinrToCapacity(30.0, 400e6, "table");
    check(approx(cap, 5.55 * 400, 0.1), "table clamps at highest MCS");
}

static void
test_table_increases_monotonically()
{
    double prev = 0.0;
    for (double sinr = -6.7; sinr <= 25.0; sinr += 0.5)
    {
        double cap = SinrToCapacity(sinr, 400e6, "table");
        check(cap >= prev, "table monotonically non-decreasing at SINR=" +
              std::to_string(sinr));
        prev = cap;
    }
}

static void
test_unknown_amc_throws()
{
    bool threw = false;
    try
    {
        SinrToCapacity(10.0, 400e6, "bogus");
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    check(threw, "unknown amc_model throws runtime_error");
}

// ---- SinrToMcsIndex tests ----

static void
test_mcs_index_below_min()
{
    check(SinrToMcsIndex(-10.0) == 0, "mcs_index below SINR_MIN returns 0");
}

static void
test_mcs_index_at_exact_thresholds()
{
    check(SinrToMcsIndex(-6.7) == 0,  "mcs_index at -6.7 dB = 0");
    check(SinrToMcsIndex(10.3) == 8,  "mcs_index at 10.3 dB = 8");
    check(SinrToMcsIndex(22.7) == 14, "mcs_index at 22.7 dB = 14");
}

static void
test_mcs_index_between_thresholds()
{
    // 3.0 dB is between index 4 (2.4) and index 5 (4.3)
    check(SinrToMcsIndex(3.0) == 4,  "mcs_index at 3.0 dB = 4");
    // 15.0 dB is between index 10 (14.1) and index 11 (16.3)
    check(SinrToMcsIndex(15.0) == 10, "mcs_index at 15.0 dB = 10");
}

static void
test_mcs_index_very_high_sinr()
{
    check(SinrToMcsIndex(50.0) == 14, "mcs_index at 50 dB clamps to 14");
}

static void
test_mcs_index_monotonic()
{
    uint32_t prev = 0;
    for (double sinr = -6.7; sinr <= 25.0; sinr += 0.5)
    {
        uint32_t idx = SinrToMcsIndex(sinr);
        check(idx >= prev, "mcs_index non-decreasing at SINR=" +
              std::to_string(sinr));
        prev = idx;
    }
}

// ---- LinkTable tests ----

static std::vector<LinkResult>
makeResults(uint32_t n)
{
    std::vector<LinkResult> results;
    for (uint32_t i = 0; i < n; ++i)
    {
        for (uint32_t j = i + 1; j < n; ++j)
        {
            LinkResult r;
            r.tx_id = i;
            r.rx_id = j;
            r.distance_m = 100.0 * (i + j + 1);
            r.is_los = (i + j) % 2 == 0;
            r.sinr_db = 20.0 - static_cast<double>(i + j) * 3.0;
            r.capacity_mbps = std::max(0.0, r.sinr_db * 50.0);
            results.push_back(r);
        }
    }
    return results;
}

static void
test_link_table_update_and_get()
{
    LinkTable table;
    auto results = makeResults(3);
    table.Update(3, results);

    check(table.NumNodes() == 3, "numNodes after update");

    // Check symmetry
    const auto& r01 = table.Get(0, 1);
    const auto& r10 = table.Get(1, 0);
    check(r01.sinr_db == r10.sinr_db, "Get(0,1) == Get(1,0) SINR");
    check(r01.distance_m == r10.distance_m, "Get(0,1) == Get(1,0) distance");
}

static void
test_link_table_wrong_result_count_throws()
{
    LinkTable table;
    auto results = makeResults(3);
    results.pop_back(); // now has 2 instead of 3

    bool threw = false;
    try
    {
        table.Update(3, results);
    }
    catch (const std::runtime_error&)
    {
        threw = true;
    }
    check(threw, "wrong result count throws");
}

static void
test_link_table_max_capacity()
{
    LinkTable table;
    auto results = makeResults(3);
    table.Update(3, results);

    double maxCap = table.MaxCapacity();
    double expectedMax = 0.0;
    for (const auto& r : results)
    {
        if (r.capacity_mbps > expectedMax)
            expectedMax = r.capacity_mbps;
    }
    check(approx(maxCap, expectedMax, 0.01), "MaxCapacity matches expected");
}

static void
test_link_table_connected_count()
{
    LinkTable table;

    // 3 nodes: links (0,1), (0,2), (1,2) with known SINRs
    std::vector<LinkResult> results(3);
    results[0].tx_id = 0; results[0].rx_id = 1; results[0].sinr_db = 10.0;
    results[1].tx_id = 0; results[1].rx_id = 2; results[1].sinr_db = -10.0;
    results[2].tx_id = 1; results[2].rx_id = 2; results[2].sinr_db = 0.0;

    table.Update(3, results);

    // Default threshold is -6.7 dB
    check(table.ConnectedLinkCount() == 2, "connected count at default threshold");
    check(table.ConnectedLinkCount(5.0) == 1, "connected count at 5 dB threshold");
    check(table.ConnectedLinkCount(-20.0) == 3, "connected count at very low threshold");
}

static void
test_link_table_is_connected()
{
    LinkTable table;
    std::vector<LinkResult> results(1);
    results[0].tx_id = 0;
    results[0].rx_id = 1;
    results[0].sinr_db = 5.0;

    table.Update(2, results);

    check(table.IsConnected(0, 1, 0.0), "link connected above threshold");
    check(!table.IsConnected(0, 1, 10.0), "link not connected below threshold");
    check(table.IsConnected(1, 0, 0.0), "symmetric IsConnected");
}

static void
test_link_table_diagonal_is_default()
{
    LinkTable table;
    std::vector<LinkResult> results(1);
    results[0].tx_id = 0;
    results[0].rx_id = 1;
    results[0].sinr_db = 10.0;
    results[0].capacity_mbps = 500.0;

    table.Update(2, results);

    const auto& diag = table.Get(0, 0);
    check(diag.capacity_mbps == 0.0, "diagonal entry has default capacity");
    check(diag.sinr_db == -999.0, "diagonal entry has default SINR");
}

// ---- main ----

int
main()
{
    // SinrToCapacity
    test_below_min_sinr_returns_zero();
    test_shannon_positive_sinr();
    test_shannon_high_sinr();
    test_shannon_increases_with_sinr();
    test_shannon_scales_with_bandwidth();
    test_table_lowest_mcs();
    test_table_mid_range();
    test_table_highest_mcs();
    test_table_increases_monotonically();
    test_unknown_amc_throws();

    // SinrToMcsIndex
    test_mcs_index_below_min();
    test_mcs_index_at_exact_thresholds();
    test_mcs_index_between_thresholds();
    test_mcs_index_very_high_sinr();
    test_mcs_index_monotonic();

    // LinkTable
    test_link_table_update_and_get();
    test_link_table_wrong_result_count_throws();
    test_link_table_max_capacity();
    test_link_table_connected_count();
    test_link_table_is_connected();
    test_link_table_diagonal_is_default();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.\n";
    if (g_fail > 0)
    {
        std::cout << "SOME TESTS FAILED.\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}

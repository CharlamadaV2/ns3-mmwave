#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CLI integration tests for mesh-sim.
#
# Usage:
#   ./tests/cli-integration-test.sh [path-to-sim-binary]
#
# If no binary is given, auto-detects from ../../build/scratch/mesh-sim/sim
# relative to the mesh-sim root.
#
# Requires the binary to be built first (the test does NOT build it).
# ---------------------------------------------------------------------------
set -euo pipefail

MESH_SIM_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -n "${1:-}" ]]; then
    BIN="$1"
else
    # Auto-detect: check common build directories for the sim binary
    REPO_ROOT="$MESH_SIM_DIR/../.."
    for candidate in \
        "$REPO_ROOT/build/scratch/mesh-sim/ns3.42-sim-debug" \
        "$REPO_ROOT/build/scratch/mesh-sim/ns3.42-sim-default" \
        "$REPO_ROOT/cmake-build-debug/scratch/mesh-sim/ns3.42-sim-debug" \
        "$REPO_ROOT/build/scratch/mesh-sim/sim" \
        "$REPO_ROOT/cmake-build-debug/scratch/mesh-sim/sim"; do
        if [[ -x "$candidate" ]]; then
            BIN="$candidate"
            break
        fi
    done
    BIN="${BIN:-$REPO_ROOT/cmake-build-debug/scratch/mesh-sim/ns3.42-sim-debug}"
fi

if [[ ! -x "$BIN" ]]; then
    echo "Error: binary not found or not executable: $BIN"
    echo "Either pass the path as an argument or build first."
    exit 1
fi

PASS=0
FAIL=0
SCRIPT_DIR="$MESH_SIM_DIR"

pass() { ((PASS++)); echo "  PASS: $1"; }
fail() { ((FAIL++)); echo "  FAIL: $1"; }

echo "Running CLI integration tests..."
echo "Binary: $BIN"
echo ""

# --- Test 1: --PrintHelp exits 0 and mentions run-config ---
echo "Test 1: --PrintHelp"
if output=$("$BIN" --PrintHelp 2>&1) && echo "$output" | grep -q "run-config"; then
    pass "--PrintHelp shows run-config"
else
    fail "--PrintHelp should exit 0 and mention run-config"
fi

# --- Test 2: Missing --run-config exits nonzero ---
echo "Test 2: missing --run-config"
if "$BIN" 2>/dev/null; then
    fail "should exit nonzero without --run-config"
else
    pass "exits nonzero without --run-config"
fi

# --- Test 3: Nonexistent config file exits nonzero ---
echo "Test 3: nonexistent config file"
if "$BIN" --run-config=/nonexistent/path/run.ini 2>/dev/null; then
    fail "should exit nonzero for nonexistent config"
else
    pass "exits nonzero for nonexistent config"
fi

# --- Test 4: Bad seed value exits nonzero ---
echo "Test 4: bad seed value"
if "$BIN" --run-config="$SCRIPT_DIR/inputs/scenarios/triangle/run.ini" --seeds=abc 2>/dev/null; then
    fail "should exit nonzero for bad seed"
else
    pass "exits nonzero for bad seed value"
fi

# --- Test 5: Nonexistent positions-override exits nonzero ---
echo "Test 5: nonexistent positions-override"
if "$BIN" --run-config="$SCRIPT_DIR/inputs/scenarios/triangle/run.ini" \
          --positions-override=/nonexistent/pos.json 2>/dev/null; then
    fail "should exit nonzero for nonexistent positions-override"
else
    pass "exits nonzero for nonexistent positions-override"
fi

# --- Test 6: Valid run produces run.log ---
echo "Test 6: valid run produces run.log"
TMPDIR_TEST=$(mktemp -d)
# Create a minimal run.ini that points to the triangle scenario files
# but outputs to our temp dir
cat > "$TMPDIR_TEST/run.ini" <<INIEOF
[scenario]
name = test-cli
seed = 1
duration_s = 0.1
tick_s = 0.1
nodes_file = $SCRIPT_DIR/inputs/scenarios/triangle/nodes.json

[channel]
frequency_ghz = 28.0
channel_model = 3gpp
scenario = UMi
bandwidth_mhz = 400.0

[traffic]
model = constant
demand_mbps = 10.0
flow_topology = all_pairs

[routing]
algorithm = shortest_path

[output]
dir = $TMPDIR_TEST/output
INIEOF

if "$BIN" --run-config="$TMPDIR_TEST/run.ini" --seed=1 >/dev/null 2>&1; then
    if [[ -f "$TMPDIR_TEST/output/run.log" ]]; then
        if grep -q "seeds:" "$TMPDIR_TEST/output/run.log" && \
           grep -q "CLI overrides:" "$TMPDIR_TEST/output/run.log"; then
            pass "run.log created with expected content"
        else
            fail "run.log exists but missing expected content"
        fi
    else
        fail "run.log not created"
    fi
else
    fail "valid run should exit 0"
fi
rm -rf "$TMPDIR_TEST"

# --- Summary ---
echo ""
echo "Results: $PASS passed, $FAIL failed."
if [[ $FAIL -gt 0 ]]; then
    echo "SOME TESTS FAILED."
    exit 1
fi
echo "All tests passed."

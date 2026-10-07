#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CLI integration tests for mesh-sim.
#
# Usage:
#   tests/integration/cli-integration-test.sh [path-to-sim-binary]
#
# Binary selection order: $MESH_SIM_BIN, then the optional positional argument,
# then the unique executable match of build/scratch/mesh-sim/ns3*-sim-*.
#
# Requires the binary to be built first (the test does NOT build it).
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MESH_SIM_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
NS3_ROOT="$(cd "$MESH_SIM_DIR/../.." && pwd)"

if [[ -n "${MESH_SIM_BIN:-}" ]]; then
    BIN="$MESH_SIM_BIN"
elif [[ -n "${1:-}" ]]; then
    BIN="$1"
else
    candidates=()
    for candidate in "$NS3_ROOT"/build/scratch/mesh-sim/ns3*-sim-*; do
        [[ -x "$candidate" ]] && candidates+=("$candidate")
    done
    if [[ ${#candidates[@]} -eq 0 ]]; then
        echo "Error: no executable matched $NS3_ROOT/build/scratch/mesh-sim/ns3*-sim-*"
        echo "Build first, pass the path as an argument, or set MESH_SIM_BIN."
        exit 1
    fi
    if [[ ${#candidates[@]} -gt 1 ]]; then
        echo "Error: multiple sim binaries matched; set MESH_SIM_BIN to pick one:"
        printf '  %s\n' "${candidates[@]}"
        exit 1
    fi
    BIN="${candidates[0]}"
fi

if [[ ! -x "$BIN" ]]; then
    echo "Error: binary not found or not executable: $BIN"
    echo "Either pass the path as an argument, set MESH_SIM_BIN, or build first."
    exit 1
fi

# ns-3 shared libraries live next to the binary's build tree.
export DYLD_LIBRARY_PATH="$NS3_ROOT/build/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$NS3_ROOT/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SMOKE="$MESH_SIM_DIR/inputs/baselines/p0-smoke"
JAMMER="$MESH_SIM_DIR/inputs/baselines/p0-jammer-smoke"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0

# Plain arithmetic assignment: ((PASS++)) returns 1 when PASS is 0, which
# would abort the run under `set -e`.
pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

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

# Tests 4-9 need the baseline fixtures; a missing file would let the negative
# tests pass for the wrong reason.
for required in \
    "$SMOKE/run.ini" "$SMOKE/nodes.json" \
    "$JAMMER/run.ini" "$JAMMER/nodes.json" "$JAMMER/jammers.json"; do
    if [[ ! -f "$required" ]]; then
        echo "Error: missing required fixture: $required"
        exit 1
    fi
done

# --- Test 4: Bad seed value is rejected with the seed parser's message ---
echo "Test 4: bad seed value"
if "$BIN" --run-config="$SMOKE/run.ini" --seeds=abc --output-dir="$TMP/run4" \
        </dev/null >/dev/null 2>"$TMP/run4.err"; then
    fail "should exit nonzero for bad seed"
elif grep -q "invalid seed value" "$TMP/run4.err"; then
    pass "exits nonzero for bad seed value with 'invalid seed value'"
else
    fail "exited nonzero but stderr lacks 'invalid seed value'"
fi

# --- Test 5: Nonexistent positions-override is reported as missing ---
echo "Test 5: nonexistent positions-override"
if "$BIN" --run-config="$SMOKE/run.ini" --positions-override=/nonexistent/pos.json \
        --output-dir="$TMP/run5" </dev/null >/dev/null 2>"$TMP/run5.err"; then
    fail "should exit nonzero for nonexistent positions-override"
elif grep -q "positions-override file not found" "$TMP/run5.err"; then
    pass "exits nonzero with 'positions-override file not found'"
else
    fail "exited nonzero but stderr lacks 'positions-override file not found'"
fi

# --- Test 6: Valid run produces run.log and per-seed outputs ---
# p0-smoke has [rl] enabled, so stdout carries the RL JSON stream and stdin is
# the action channel; closed stdin makes every action a Stay.
echo "Test 6: valid run produces run.log"
if "$BIN" --run-config="$SMOKE/run.ini" --seed=1 --output-dir="$TMP/run6" \
        </dev/null >/dev/null 2>"$TMP/run6.err"; then
    if [[ ! -f "$TMP/run6/run.log" ]]; then
        fail "run.log not created"
    elif ! grep -q "seeds:" "$TMP/run6/run.log" || \
         ! grep -q "CLI overrides:" "$TMP/run6/run.log"; then
        fail "run.log exists but missing expected content"
    elif [[ ! -f "$TMP/run6/seed-1/summary.json" ]]; then
        fail "seed-1/summary.json not created"
    elif [[ ! -f "$TMP/run6/seed-1/links.csv" ]]; then
        fail "seed-1/links.csv not created"
    else
        pass "run.log, summary.json and links.csv created"
    fi
else
    fail "valid run should exit 0"
fi

# --- Test 7: --band on the CLI overrides the scenario band ---
echo "Test 7: CLI band precedence"
if "$BIN" --run-config="$SMOKE/run.ini" --band=sub-6 --seed=1 \
        --output-dir="$TMP/run7" </dev/null >/dev/null 2>&1; then
    if [[ ! -f "$TMP/run7/run.log" ]]; then
        fail "run.log not created for --band override"
    elif grep -Eq '^  band +=  *sub-6$' "$TMP/run7/run.log" && \
         grep -Eq '^  band_source +=  *cli$' "$TMP/run7/run.log"; then
        pass "run.log records band = sub-6 and band_source = cli"
    else
        fail "run.log does not record band = sub-6 with band_source = cli"
    fi
else
    fail "run with --band=sub-6 should exit 0"
fi

# --- Test 8: jammer scenario differs between sub-6 and mmwave ---
echo "Test 8: jammer A/B across bands"
jam_ok=1
if ! "$BIN" --run-config="$JAMMER/run.ini" --band=sub-6 --seed=1 \
        --output-dir="$TMP/jam-sub6" </dev/null >/dev/null 2>&1; then
    fail "jammer run with --band=sub-6 should exit 0"
    jam_ok=0
fi
if ! "$BIN" --run-config="$JAMMER/run.ini" --band=mmwave --seed=1 \
        --output-dir="$TMP/jam-mmwave" </dev/null >/dev/null 2>&1; then
    fail "jammer run with --band=mmwave should exit 0"
    jam_ok=0
fi

if [[ $jam_ok -eq 1 ]]; then
    if ! grep -Eq 'jammer_path_enabled *= *true' "$TMP/jam-sub6/run.log" 2>/dev/null; then
        fail "sub-6 run.log should record jammer_path_enabled = true"
    elif ! grep -Eq 'jammer_path_enabled *= *false' "$TMP/jam-mmwave/run.log" 2>/dev/null; then
        fail "mmwave run.log should record jammer_path_enabled = false"
    elif [[ ! -f "$TMP/jam-sub6/seed-1/links.csv" || ! -f "$TMP/jam-mmwave/seed-1/links.csv" ]]; then
        fail "both jammer runs should produce seed-1/links.csv"
    else
        # Match rows on (time_s,node_a,node_b) and count sinr_db differences.
        cmp_out=$(awk -F, '
            function strip(s) { gsub(/\r/, "", s); return s }
            FNR == 1 { file++ }
            /^#/ { next }
            !hdr[file] {
                for (i = 1; i <= NF; i++) {
                    k = strip($i)
                    if (k == "sinr_db") sinr[file] = i
                    if (k == "time_s")  t[file] = i
                    if (k == "node_a")  a[file] = i
                    if (k == "node_b")  b[file] = i
                }
                hdr[file] = 1
                next
            }
            {
                key = strip($(t[file])) "," strip($(a[file])) "," strip($(b[file]))
                if (file == 1) { v1[key] = strip($(sinr[file])); n1++ }
                else           { v2[key] = strip($(sinr[file])); n2++ }
            }
            END {
                diff = 0; missing = 0
                for (k in v1) {
                    if (!(k in v2)) missing++
                    else if (v1[k] != v2[k]) diff++
                }
                for (k in v2) if (!(k in v1)) missing++
                print diff, missing, n1, n2
            }
        ' "$TMP/jam-sub6/seed-1/links.csv" "$TMP/jam-mmwave/seed-1/links.csv")
        read -r n_diff n_missing rows1 rows2 <<<"$cmp_out"
        if [[ "$n_missing" -ne 0 || "$rows1" -ne "$rows2" ]]; then
            fail "links.csv key sets differ ($n_missing unmatched, $rows1 vs $rows2 rows)"
        elif [[ "$n_diff" -ge 1 ]]; then
            pass "jammer band A/B differs in $n_diff matched links"
        else
            fail "no matched link differs in sinr_db between sub-6 and mmwave"
        fi
    fi
fi

# --- Test 9: legacy stream shape is unchanged ---
echo "Test 9: legacy RL stream shape"
if "$BIN" --run-config="$SMOKE/run.ini" --seed=1 --output-dir="$TMP/run12" \
        </dev/null >"$TMP/run12.out" 2>"$TMP/run12.err"; then
    line_count=$(grep -c '' "$TMP/run12.out" || true)
    if [[ "$line_count" -ne 5 ]]; then
        fail "legacy stdout should have 5 lines, got $line_count"
    elif grep -q '"type":"init"' "$TMP/run12.out"; then
        fail "legacy stdout must not contain an init message"
    elif ! head -n 1 "$TMP/run12.out" | grep -q 'controlled_pos'; then
        fail "legacy first line should contain controlled_pos"
    else
        pass "legacy stream is 5 step lines with no init message"
    fi
else
    fail "legacy p0-smoke run should exit 0"
fi

# Temporary p0-smoke copies with a [baseline] section; <name> <rl-enabled> <algorithm>.
make_baseline_scenario() {
    local dir="$TMP/$1"
    mkdir -p "$dir"
    cp "$SMOKE/nodes.json" "$dir/nodes.json"
    sed -E "s/^enabled( *)= *true\$/enabled\\1= $2/" "$SMOKE/run.ini" > "$dir/run.ini"
    # Avoid auto-output root discovery for an INI copied outside mesh-sim.
    printf 'dir = .\n\n[baseline]\nalgorithm = %s\n' "$3" >> "$dir/run.ini"
    if ! grep -Eq "^enabled *= *$2\$" "$dir/run.ini"; then
        echo "Error: could not set [rl] enabled = $2 in $dir/run.ini"
        exit 1
    fi
}

make_baseline_scenario bl-geo-rl-off false geometric
make_baseline_scenario bl-geo-rl-on true geometric
make_baseline_scenario bl-none-rl-off false none

LAUNCHER_HINT="python -m scripts.baselines.runner --run-config <ini>"
RL_NOTICE="Note: [baseline] algorithm 'geometric' is not applied in RL mode"

# --- Test 10: active baseline without --rl-mode is refused ---
echo "Test 10: active baseline refused on a direct run"
for variant in bl-geo-rl-off bl-geo-rl-on; do
    if "$BIN" --run-config="$TMP/$variant/run.ini" --seed=1 \
            --output-dir="$TMP/run10-$variant" </dev/null >/dev/null 2>"$TMP/run10-$variant.err"; then
        fail "$variant: direct run with an active baseline should exit nonzero"
    elif ! grep -qF "$LAUNCHER_HINT" "$TMP/run10-$variant.err"; then
        fail "$variant: exited nonzero but stderr lacks the launcher hint"
    elif [[ -e "$TMP/run10-$variant" ]]; then
        fail "$variant: refused run should not create its output directory"
    else
        pass "$variant: refused with the launcher hint and no output"
    fi
done

# --- Test 11: algorithm = none runs as before ---
echo "Test 11: baseline algorithm = none runs"
if "$BIN" --run-config="$TMP/bl-none-rl-off/run.ini" --seed=1 \
        --output-dir="$TMP/run11" </dev/null >/dev/null 2>"$TMP/run11.err"; then
    if [[ ! -f "$TMP/run11/seed-1/summary.json" ]]; then
        fail "algorithm = none run did not create seed-1/summary.json"
    elif grep -q "\[baseline\]" "$TMP/run11.err"; then
        fail "algorithm = none run should print no baseline notice"
    else
        pass "algorithm = none runs and prints no baseline notice"
    fi
else
    fail "algorithm = none run should exit 0"
fi

# --- Test 12: active baseline with explicit --rl-mode runs with a notice ---
echo "Test 12: active baseline with --rl-mode"
if "$BIN" --run-config="$TMP/bl-geo-rl-off/run.ini" --rl-mode --seed=1 \
        --output-dir="$TMP/run12b" </dev/null >/dev/null 2>"$TMP/run12b.err"; then
    if ! grep -qF "$RL_NOTICE" "$TMP/run12b.err"; then
        fail "--rl-mode run with an active baseline should print the notice"
    elif [[ ! -f "$TMP/run12b/seed-1/summary.json" ]]; then
        fail "--rl-mode run did not create seed-1/summary.json"
    else
        pass "--rl-mode run starts and prints the baseline notice"
    fi
else
    fail "--rl-mode run with an active baseline should exit 0"
fi

# --- Channel query (--channel-query, contract mesh_channel_query_v1) ---
# JSON checks use python3's standard library; any assertion message is printed on failure.
RWALK="$MESH_SIM_DIR/inputs/baselines/16-random-walk-urban"
for required in "$RWALK/run.ini" "$RWALK/nodes.json" "$RWALK/buildings.json"; do
    if [[ ! -f "$required" ]]; then
        echo "Error: missing required fixture: $required"
        exit 1
    fi
done
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required for the channel query tests"
    exit 1
fi

JAMMER_START='[[0.0, 0.0, 10.0], [100.0, 0.0, 10.0], [50.0, 50.0, 10.0]]'

# run_bounded <seconds> <stdin> <stdout> <stderr> <cmd...>: the command's status,
# or 124 after terminating (then killing) a command that overran the bound.
run_bounded() {
    local limit="$1" in="$2" out="$3" err="$4"
    shift 4
    "$@" <"$in" >"$out" 2>"$err" &
    local pid=$! ticks=0 rc=0
    while kill -0 "$pid" 2>/dev/null; do
        if [[ $ticks -ge $((limit * 10)) ]]; then
            kill -TERM "$pid" 2>/dev/null || true
            sleep 1
            kill -KILL "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            return 124
        fi
        sleep 0.1
        ticks=$((ticks + 1))
    done
    wait "$pid" || rc=$?
    return "$rc"
}

# --- Test 13: init line on p0-jammer-smoke with --band=sub-6 ---
echo "Test 13: channel query init line"
rc=0
run_bounded 60 /dev/null "$TMP/q13.out" "$TMP/q13.err" \
    "$BIN" --run-config="$JAMMER/run.ini" --band=sub-6 --channel-query --seed=3 \
    --output-dir="$TMP/q13-out" || rc=$?
if [[ $rc -ne 0 ]]; then
    fail "query worker with empty stdin should exit 0 (got $rc)"
elif [[ -e "$TMP/q13-out" ]]; then
    fail "query mode must not create the output directory"
elif ! grep -qF -- "--output-dir is ignored with --channel-query" "$TMP/q13.err"; then
    fail "stderr lacks the --output-dir note"
elif msg=$(python3 - "$TMP/q13.out" "$JAMMER_START" 2>&1 <<'PY'
import json, sys
lines = open(sys.argv[1]).read().splitlines()
assert len(lines) == 1, f"expected only the init line, got {len(lines)} lines"
m = json.loads(lines[0])
expect = {"type": "init", "contract": "mesh_channel_query_v1", "isolation": "fork_per_layout",
          "node_ids": ["node-a", "node-b", "relay"], "band": "sub-6", "band_source": "cli",
          "seed": 3, "run_id": 1, "jammer_seed": 3, "sinr_threshold_db": -6.7,
          "jammer_path_enabled": True, "num_buildings": 0, "rl_enabled": False,
          "controlled_indices": [], "time_s": 0.0}
for key, value in expect.items():
    assert m.get(key) == value, f"{key}: {m.get(key)!r} != {value!r}"
assert m["start_positions"] == json.loads(sys.argv[2]), m["start_positions"]
assert m["limits"]["max_layouts"] == 1024 and m["limits"]["max_probes"] == 10000, m["limits"]
assert m["channel"]["frequency_ghz"] == 2.4 and m["channel"]["condition_model"] == "static_los"
PY
); then
    pass "init line carries the contract, roster, seeds, band and limits"
else
    fail "init line check failed: $msg"
fi

# --- Test 14: one layout returns 3 links; EOF exits 0 ---
echo "Test 14: one-layout channel query"
printf '{"type":"evaluate","request_id":11,"layouts":[%s]}\n' "$JAMMER_START" > "$TMP/q14.in"
rc=0
run_bounded 60 "$TMP/q14.in" "$TMP/q14.out" "$TMP/q14.err" \
    "$BIN" --run-config="$JAMMER/run.ini" --band=sub-6 --channel-query --seed=1 || rc=$?
if [[ $rc -ne 0 ]]; then
    fail "query worker should exit 0 at EOF (got $rc)"
elif msg=$(python3 - "$TMP/q14.out" 2>&1 <<'PY'
import json, math, sys
lines = [json.loads(l) for l in open(sys.argv[1]).read().splitlines()]
assert len(lines) == 2, f"expected init + 1 result, got {len(lines)} lines"
r = lines[1]
assert r["type"] == "result" and r["request_id"] == 11, r
assert len(r["layouts"]) == 1, r
L = r["layouts"][0]
assert "error" not in L, L
assert L["coverage"] is None, L["coverage"]
links = L["links"]
assert [row[:2] for row in links] == [[0, 1], [0, 2], [1, 2]], links
for i, j, sinr, cap, los, connected in links:
    assert math.isfinite(sinr) and math.isfinite(cap), (sinr, cap)
    assert isinstance(los, bool) and connected == (sinr >= -6.7), (sinr, connected)
assert L["wall_s"] >= 0 and r["wall_s"] >= 0
PY
); then
    pass "3 links in i<j order with connected == (sinr_db >= -6.7); EOF exits 0"
else
    fail "one-layout result check failed: $msg"
fi

# --- Test 15: request errors keep the worker serving; shutdown stops it ---
echo "Test 15: malformed request then valid request"
{
    echo '{"type":"evaluate","layouts":[[[0,0,10]'
    echo '{"type":"bogus","request_id":5}'
    printf '{"type":"evaluate","request_id":6,"layouts":[%s]}\n' "$JAMMER_START"
    echo '{"type":"shutdown"}'
    printf '{"type":"evaluate","request_id":7,"layouts":[%s]}\n' "$JAMMER_START"
} > "$TMP/q15.in"
rc=0
run_bounded 60 "$TMP/q15.in" "$TMP/q15.out" "$TMP/q15.err" \
    "$BIN" --run-config="$JAMMER/run.ini" --band=sub-6 --channel-query --seed=1 || rc=$?
if [[ $rc -ne 0 ]]; then
    fail "query worker should exit 0 on shutdown (got $rc)"
elif msg=$(python3 - "$TMP/q15.out" 2>&1 <<'PY'
import json, sys
lines = [json.loads(l) for l in open(sys.argv[1]).read().splitlines()]
types = [(m["type"], m.get("request_id")) for m in lines]
assert types == [("init", None), ("error", None), ("error", 5), ("result", 6)], types
assert "malformed" in lines[1]["message"] and "bogus" in lines[2]["message"], lines[1:3]
assert "error" not in lines[3]["layouts"][0], lines[3]
PY
); then
    pass "malformed and unknown requests answered with errors; worker kept serving until shutdown"
else
    fail "error/serving check failed: $msg"
fi

# --- Test 16: query mode needs exactly one seed ---
echo "Test 16: --channel-query with two seeds"
rc=0
run_bounded 60 /dev/null "$TMP/q16.out" "$TMP/q16.err" \
    "$BIN" --run-config="$JAMMER/run.ini" --channel-query --seeds=1,2 || rc=$?
if [[ $rc -ne 1 ]]; then
    fail "two seeds should exit 1 (got $rc)"
elif ! grep -qF "Error: --channel-query needs exactly one seed" "$TMP/q16.err"; then
    fail "stderr lacks the one-seed error"
elif [[ -s "$TMP/q16.out" ]]; then
    fail "stdout should be empty when the seed check fails"
else
    pass "two seeds rejected with the one-seed error"
fi

# --- Test 17: a layout outside random-walk bounds is a layout error, not an abort ---
echo "Test 17: random-walk layout outside bounds"
printf '%s\n' '{"type":"evaluate","request_id":1,"layouts":[[[150,0,30],[500,50,1.5],[250,-30,1.5]],[[150,0,30],[50,50,1.5],[250,-30,1.5]]]}' \
    > "$TMP/q17.in"
rc=0
run_bounded 60 "$TMP/q17.in" "$TMP/q17.out" "$TMP/q17.err" \
    "$BIN" --run-config="$RWALK/run.ini" --channel-query || rc=$?
if [[ $rc -ne 0 ]]; then
    fail "query worker should exit 0 (got $rc)"
elif msg=$(python3 - "$TMP/q17.out" 2>&1 <<'PY'
import json, sys
lines = [json.loads(l) for l in open(sys.argv[1]).read().splitlines()]
assert len(lines) == 2 and lines[1]["type"] == "result", lines
bad, good = lines[1]["layouts"]
assert "random_walk bounds" in bad.get("error", ""), bad
assert "error" not in good and len(good["links"]) == 3, good
PY
); then
    pass "outside-bounds layout reported per layout; the next layout still scored"
else
    fail "random-walk layout check failed: $msg"
fi

# --- Test 18: a response larger than the pipe capacity drains without deadlock ---
echo "Test 18: large probe response"
python3 - "$JAMMER_START" > "$TMP/q18.in" <<'PY'
import json, sys
points = [[float(x), float(y)] for x in range(100) for y in range(-50, 50)]
print(json.dumps({"type": "evaluate", "request_id": 18, "layouts": [json.loads(sys.argv[1])],
                  "probes": {"height_m": 1.5, "rx_gain_dbi": 12.0, "sinr_db": -6.7,
                             "points": points}}))
PY
rc=0
run_bounded 300 "$TMP/q18.in" "$TMP/q18.out" "$TMP/q18.err" \
    "$BIN" --run-config="$JAMMER/run.ini" --band=sub-6 --channel-query --seed=1 || rc=$?
if [[ $rc -eq 124 ]]; then
    fail "large probe request did not finish within 300 s"
elif [[ $rc -ne 0 ]]; then
    fail "query worker should exit 0 (got $rc)"
elif msg=$(python3 - "$TMP/q18.out" 2>&1 <<'PY'
import json, sys
lines = open(sys.argv[1]).read().splitlines()
assert len(lines) == 2, f"expected init + 1 result, got {len(lines)} lines"
r = json.loads(lines[1])
L = r["layouts"][0]
assert "error" not in L, L
assert len(L["coverage"]) == 3, L["coverage"]
for covered in L["coverage"]:
    assert covered == sorted(set(covered)) and all(0 <= k < 10000 for k in covered)
assert len(lines[1]) > 65536, f"response is only {len(lines[1])} bytes"
PY
); then
    pass "response over 64 KiB drained and parsed"
else
    fail "large response check failed: $msg"
fi

# --- Test 19: SIGTERM to a busy worker leaves no query child behind ---
# Only the worker's own PID, its children and (with job control) its process group are inspected.
echo "Test 19: worker termination cleans up its child"
python3 - "$JAMMER_START" > "$TMP/q19.in" <<'PY'
import json, sys
points = [[float(x), float(y)] for x in range(100) for y in range(-50, 50)]
layout = json.loads(sys.argv[1])
print(json.dumps({"type": "evaluate", "request_id": 19, "layouts": [layout] * 1024,
                  "probes": {"height_m": 1.5, "rx_gain_dbi": 12.0, "sinr_db": -6.7,
                             "points": points}}))
PY
set -m
"$BIN" --run-config="$JAMMER/run.ini" --band=sub-6 --channel-query --seed=1 \
    <"$TMP/q19.in" >"$TMP/q19.out" 2>"$TMP/q19.err" &
qpid=$!
set +m
qpgid=$(ps -o pgid= -p "$qpid" 2>/dev/null | tr -d ' ' || true)
child=""
for _ in $(seq 1 200); do
    child=$(pgrep -P "$qpid" 2>/dev/null | head -n 1 || true)
    [[ -n "$child" ]] && break
    sleep 0.1
done
# Re-read the active child: layouts run one child at a time.
active=$(pgrep -P "$qpid" 2>/dev/null | head -n 1 || true)
[[ -n "$active" ]] && child="$active"
kill -TERM "$qpid" 2>/dev/null || true
for _ in $(seq 1 100); do
    kill -0 "$qpid" 2>/dev/null || break
    sleep 0.1
done
kill -KILL "$qpid" 2>/dev/null || true
wait "$qpid" 2>/dev/null || true
leftover=""
for _ in $(seq 1 50); do
    leftover=""
    if [[ -n "$child" ]]; then
        st=$(ps -o stat= -p "$child" 2>/dev/null || true)
        [[ -n "$st" && "$st" != *Z* ]] && leftover="child $child"
    fi
    if [[ "$qpgid" == "$qpid" ]]; then
        group=$(pgrep -g "$qpid" 2>/dev/null | tr '\n' ' ' || true)
        [[ -n "$group" ]] && leftover="$leftover group: $group"
    fi
    [[ -z "$leftover" ]] && break
    sleep 0.1
done
if [[ -z "$child" ]]; then
    fail "no query child appeared within 20 s"
elif [[ -n "$leftover" ]]; then
    fail "processes remain after terminating the worker: $leftover"
    [[ "$qpgid" == "$qpid" ]] && kill -KILL -- "-$qpid" 2>/dev/null || true
    kill -KILL "$child" 2>/dev/null || true
else
    pass "SIGTERM to the worker also removed its active child"
fi

# --- Summary ---
echo ""
echo "Results: $PASS passed, $FAIL failed."
if [[ $FAIL -gt 0 ]]; then
    echo "SOME TESTS FAILED."
    exit 1
fi
echo "All tests passed."

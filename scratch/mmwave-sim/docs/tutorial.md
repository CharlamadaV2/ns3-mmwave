# Tutorial

Step-by-step walkthroughs for each scenario.

---

## 1. Validation scenarios

Run these before novel experiments to verify the simulator is behaving
correctly.  Each tests one specific physical mechanism.

### 1a. ue-moving-away — SINR degrades with distance

```bash
python3 scratch/mmwave-sim/scripts/run_sim.py \
    --config scratch/mmwave-sim/inputs/scenarios/ue-moving-away/run.ini
```

Open `RxPacketTrace.txt` (or `summary.json` per_ue) and confirm that:
- Mean SINR is high early in the run and falls over time
- `MmWaveSinrTime.txt` shows a monotone decrease

You can also open `links.csv` to see the distance increasing and `sinr_db`
decreasing over the 10-second simulation.

### 1b. two-ues-one-approaching — differential SINR improvement

```bash
python3 scratch/mmwave-sim/scripts/run_sim.py \
    --config scratch/mmwave-sim/inputs/scenarios/two-ues-one-approaching/run.ini
```

In `summary.json`, confirm:
- `per_ue.ue1.mean_sinr_db` > `per_ue.ue0.mean_sinr_db`
  (ue1 approaches from 150 m; ue0 stays at 100 m)
- `per_ue.ue1.dl_throughput_mbps` > `per_ue.ue0.dl_throughput_mbps`

### 1c. building-bypass — LOS/NLOS transition

```bash
python3 scratch/mmwave-sim/scripts/run_sim.py \
    --config scratch/mmwave-sim/inputs/scenarios/building-bypass/run.ini
```

In `links.csv`, find rows for node_a=0 (eNB (evolved Node B)) and node_b=1
(ue0).  The `condition` column should change from `NLOS` (Non-Line of Sight)
to `LOS` (Line of Sight) around t≈2 s as ue0 moves out from behind the
building.  The `sinr_db` column should increase by ~10–20 dB at that
transition — SINR (Signal-to-Interference-plus-Noise Ratio) improves as path
loss drops when the UE (User Equipment) clears the blockage.  This value comes
directly from the simulation's RRC (Radio Resource Control) measurement reports, it is not estimated.

### 1d. building-two-ues — differential blockage

```bash
python3 scratch/mmwave-sim/scripts/run_sim.py \
    --config scratch/mmwave-sim/inputs/scenarios/building-two-ues/run.ini
```

In `summary.json`, confirm:
- `per_ue.ue0.outage_fraction` > `per_ue.ue1.outage_fraction`
  (ue0 stays permanently NLOS; ue1 exits the building shadow at ~t=2 s)
- `per_ue.ue1.mean_sinr_db` > `per_ue.ue0.mean_sinr_db`

---

## 2. Modifying traffic load

All traffic parameters live in `[traffic]` in `run.ini`.

```ini
# Light load — 11.2 Mbps per UE
inter_packet_interval_us = 1000.0

# Heavy load — 224 Mbps per UE (likely saturates the scheduler)
inter_packet_interval_us = 50.0

# Enable uplink traffic in addition to downlink
direction = both

# Cap traffic at 1000 packets per source (useful for delivery-ratio tests)
max_packets = 1000
```

---

## 3. Enabling PCAP captures

Add to `[output]` in `run.ini`:
```ini
pcap_enabled = true
```

PCAP files are written to `<output_dir>/pcap/backhaul-*.pcap`.
Open with Wireshark or `tcpdump -r <file>`.

---

## 4. Adding a new scenario

1. Copy an existing scenario directory:
   ```bash
   cp -r scratch/mmwave-sim/inputs/scenarios/single-enb-3ue \
         scratch/mmwave-sim/inputs/scenarios/my-scenario
   ```

2. Edit `run.ini` — set `name`, adjust channel and traffic parameters.

3. Edit `nodes.json` — add/remove/reposition nodes.  See `docs/input-schema.md`
   for all `role`, `mobility`, and `random_walk` options.

4. Edit `buildings.json` — add obstacles.  Use `[]` for no buildings.

5. Run:
   ```bash
   python3 scratch/mmwave-sim/scripts/run_sim.py \
       --config scratch/mmwave-sim/inputs/scenarios/my-scenario/run.ini
   ```

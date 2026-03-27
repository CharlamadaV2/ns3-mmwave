# setup/

NS-3 object wiring -- topology construction, traffic installation, and protocol defaults. This is the only layer that creates ns-3 simulation objects.

## Files

### topology-builder.h / topology-builder.cc

Creates the full network topology from `SimConfig`.

**Key functions:**
- `TopologyBuilder::ConfigureChannelDefaults(cfg)` -- static. Sets `Config::SetDefault` for channel/pathloss models. **Must** be called before `CreateHelpers()`.
- `TopologyBuilder::CreateHelpers(cfg)` -- static. Returns `MmWaveHelpers` (MmWaveHelper + EPC helper).
- `TopologyBuilder::Build()` -- orchestrates the full build sequence:
  1. Configure frequency via component carrier params
  2. Create eNB and UE nodes
  3. Install mobility models (fixed, constant velocity, random walk)
  4. Create building obstacles and install `BuildingsHelper`
  5. Install mmWave devices
  6. Wire EPC: remote host, P2P backhaul, static routing
  7. Assign UE IP addresses, attach to closest eNB

**Accessors** (used by `TrafficSetup` and `VizWriter`):
- `GetEnbNodes()`, `GetUeNodes()`, `GetRemoteHost()`, `GetRemoteHostAddr()`
- `GetUeIpInterfaces()`, `GetChannelConditionModel()`

### traffic-setup.h / traffic-setup.cc

Installs UDP client/sink application pairs.

**Key functions:**
- `TrafficSetup(cfg, topology)` -- constructor
- `Install()` -- installs apps based on `cfg.traffic.direction`:
  - `"dl"` -- `UdpClient` on remote host, `PacketSink` on each UE
  - `"ul"` -- `UdpClient` on each UE, `PacketSink` on remote host
  - `"both"` -- both directions

### ns3-defaults.h / ns3-defaults.cc

Centralized `Config::SetDefault` calls.

**Key functions:**
- `ApplyProtocolDefaults()` -- fixed RLC/HARQ/buffer defaults. Call once before seed loop.
- `ApplyTuningDefaults(cfg)` -- config-derived defaults (channel update periods, CQI, AMC). Call once before seed loop.
- `ApplyTraceFileDefaults(output_dir)` -- per-seed trace file path redirection. Call before helper construction each seed.

## Usage

In `sim.cc`:
```cpp
// Once, before seed loop:
mmwave_sim::ApplyProtocolDefaults();
mmwave_sim::ApplyTuningDefaults(cfg);

// Per seed:
mmwave_sim::ApplyTraceFileDefaults(cfg.output_dir);
mmwave_sim::TopologyBuilder::ConfigureChannelDefaults(cfg);
auto [mmwH, epcH] = mmwave_sim::TopologyBuilder::CreateHelpers(cfg);
mmwave_sim::TopologyBuilder topology(cfg, mmwH, epcH);
topology.Build();
mmwave_sim::TrafficSetup traffic(cfg, topology);
traffic.Install();
```

## Notes

- Channel model selection (3gpp vs NYU) happens inside `ConfigureChannel()` based on `cfg.channel.channel_model`. The 3gpp and NYU paths set different ns-3 TypeId defaults.
- `ConfigureChannelDefaults()` must run before `CreateObject<MmWaveHelper>()` so the helper picks up the correct models at construction time.

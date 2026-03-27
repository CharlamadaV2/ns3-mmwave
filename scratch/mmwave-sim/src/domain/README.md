# domain/

Pure POD data types shared across the entire project. No logic, no ns-3 dependency. This is the dependency firewall -- every other module depends on domain, but domain depends on nothing.

## Files

### sim-config.h

Top-level simulation configuration and runtime metadata.

**Key types:**
- `SimConfig` -- central config struct (scenario name, seed, duration, warmup, output dir, channel, traffic, network, nodes, buildings, viz/pcap/trace settings)
- `TrafficConfig` -- traffic direction, packet size, inter-packet interval, max packets
- `NetworkConfig` -- backhaul data rate and delay (core network, not RF)
- `TimingInfo` -- wall-clock start/end/elapsed, populated by `sim.cc` after each run

### node-spec.h

Topology and geometry types.

**Key types:**
- `Position` -- (x, y, z) in metres
- `Velocity` -- (vx, vy, vz) in m/s
- `RandomWalkParams` -- bounding box and speed for random walk mobility
- `NodeSpec` -- id, role (enb/ue), mobility type, position, velocity, random walk params
- `BuildingSpec` -- id, axis-aligned bounding box, building type, exterior walls, floors

### channel-config.h

Radio and channel model parameters.

**Key types:**
- `ChannelConfig` -- frequency, tx power, scenario, channel model (3gpp/nyu), blockage, update periods, beamforming, CQI, AMC
- `NyuChannelConfig` -- NYU-specific params (bandwidth, shadowing, atmosphere, foliage, O2I loss)

## Notes

- All structs use default-initialized fields so partial construction is safe.
- `NyuChannelConfig` fields are ignored when `channel_model = "3gpp"`.

# mesh-sim

Lightweight time-stepped mmWave mesh simulator built on ns3-mmwave.

## Build

All commands run from the **ns3-mmwave repo root** (two levels above this directory).

```bash
./ns3 clean
./ns3 configure --build-profile=debug -- -DCMAKE_OSX_ARCHITECTURES=arm64
./ns3 build
```

If `./ns3 clean` doesn't clear the cache fully, remove it manually first:

```bash
rm -rf cmake-cache build
```

## Run

### Single scenario

```bash
./build/scratch/mesh-sim/ns3*-sim-* \
  --run-config=scratch/mesh-sim/inputs/baselines/01-static-los-baseline/run.ini
```

### Sweep

```bash
python -m scripts.sweep.cli --config inputs/custom/sherpa/1.1/sweep.ini
```

## Documentation
The documentation of this module is available at @c file:///C:/Users/[Your Name]/[Repository File Path]/ns3-mmwave/scratch/mesh-sim/docs/html/index.html  
This link is accessable through your favorite brower.

## About
This sim is being worked on by the University of Massachusetts's ACNL

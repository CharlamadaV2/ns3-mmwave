@page src_setup src/setup

@brief Files for generating buildings, nodes, and propogration model.  

Based on parameters and @ref sim-config.h "simulation config", the topology builder constructs ns-3 nodes, mobility 
models, propagration models, and buildings accordingly.

## Output

When sim config is passed in parameter, topology data structure is generated.

## Module Layout

| File Name | Description |
| -- | -- |
@ref topology-builder.h "topology-builder.h" | Contains information on function and class behavior.
@ref topology-builder.cc "topology-builder.cc" | Logic of how Topology builder interacts with simulation config.

## Conventions 
- Have topology-builder behaviors be based on sim configuration.
- Comment convention will enforce JavaDoc like styling (/** */). 

## Dependencies

No dependencies
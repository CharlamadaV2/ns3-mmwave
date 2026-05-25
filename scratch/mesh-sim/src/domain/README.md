# src/domain
@brief Configuration headers to be used by @ref config-loader.cc.  

The domain folder contains the configuration for classes and objects used in the simulation.
The @ref sim-config.h is contains main config struct which defines the variables and objects
configurations inside the simulation. 

## Module Layout

| File Name | Description |
| -- | -- |
@ref channel-config.h "channel-config.h" | Radio config for model selection and propogation variables.
@ref link-result.h "link-result.h" | Snapshot of the radio conditions.
@ref mesh-config.h "mesh-config.h" | Traffic generation and routing algorithm config.
@ref node-spec.h "node-spec.h" | Config for node positioning, velocity, and mobility models.
@ref sim-config.h "sim-config.h" | Root config struct. Aggregates everything below it plus runtime metadata.


## Conventions 
- Declare config structs inside @ref sim-config, while defining their parameters in their respective header. 
- Comment convention will enforce JavaDoc like styling (/** */). 

## Dependencies

No dependencies 

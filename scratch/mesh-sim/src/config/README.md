@page src_config src/config

@brief Contains scripts for loading the simulation's configuration files and config validators.  

- The @ref config-validator.h "validator" checks whether the domains' configuration file falls under the rule.
- The @ref config-loader.h "loader" parses through the JSON files and @c run.ini contained in inputs/baselines for the scenarios

## Output

- The config loader populates SimConfig object with scenarios config parameters
- The config validator returns list of errors and boolean for success

The outputs are passed into their respective data structures

## Module Layout

| File Name | Description |
| -- | -- |
@ref config-loader.h "config-loader.h" | Contains class for simulation config loader 
@ref config-loader.cc "config-loader.cc" | Logic for parsing through scenario object parameters.
@ref config-validator.h "config-validator.h" | Contains struct for validation boolean.
@ref config-validator.cc "config-validator.cc" | Contains logic for resultant @c errors on failed checks.


## Conventions
 
- Validation checks should only check for failures and perform @c push_back for errors.
- Read to SimConfig object @c cfg for loading configuration parameters. 
- Comment convention will enforce JavaDoc like styling (/** */). 

## Dependencies

Requires "third_party/json.hpp" to run
/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#pragma once

#include "src/domain/node-spec.h"
#include "third_party/json.hpp"

namespace mmwave_sim
{

/** Parse a single NodeSpec from a JSON object. */
NodeSpec parseNodeSpec(const nlohmann::json& j);

/** Parse a single BuildingSpec from a JSON object. */
BuildingSpec parseBuildingSpec(const nlohmann::json& j);

}  // namespace mmwave_sim

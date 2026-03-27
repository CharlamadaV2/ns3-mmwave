/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#pragma once

#include <map>
#include <string>

namespace mmwave_sim
{

/** Section -> (key -> value) map produced by parseIni(). */
using IniMap = std::map<std::string, std::map<std::string, std::string>>;

/**
 * Parse an INI-style config file.
 * Supports [section] headers, key=value pairs, and # / ; comments.
 */
IniMap parseIni(const std::string& path);

/** Retrieve a string value from an IniMap, returning @p def if not found. */
std::string iniGet(const IniMap& ini,
                   const std::string& section,
                   const std::string& key,
                   const std::string& def = "");

/** Retrieve a boolean value from an IniMap, returning @p def if not found. */
bool iniGetBool(const IniMap& ini,
                const std::string& section,
                const std::string& key,
                bool def);

}  // namespace mmwave_sim

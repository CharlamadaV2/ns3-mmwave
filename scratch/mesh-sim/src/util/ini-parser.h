/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#pragma once

#include <map>
#include <string>

namespace mesh_sim
{

/** @brief
*/
/** Section -> (key -> value) map produced by parseIni(). */
using IniMap = std::map<std::string, std::map<std::string, std::string>>;

/**
 * Parse an INI-style config file.
 * Supports [section] headers, key=value pairs, and # / ; comments.
 */
 /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
IniMap parseIni(const std::string& path);

/** Retrieve a string value from an IniMap, returning @p def if not found. */
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
std::string iniGet(const IniMap& ini,
                   const std::string& section,
                   const std::string& key,
                   const std::string& def = "");

/** Retrieve a boolean value from an IniMap, returning @p def if not found. */
/**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
bool iniGetBool(const IniMap& ini,
                const std::string& section,
                const std::string& key,
                bool def);

}  // namespace mesh_sim

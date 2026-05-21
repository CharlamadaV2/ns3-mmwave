/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file ini-parser.h
 * @brief Minimal INI-file parser used by @ref ConfigLoader.
 *
 * **Format rules**
 * - Comments: everything from the first @c # or @c ; to the end of the line
 *   is stripped before processing.
 * - Section headers: @c [name] — the section name is trimmed of whitespace.
 * - Key-value pairs: @c key=value — split on the *first* @c = sign; both
 *   key and value are trimmed of whitespace.
 * - Lines without an @c = sign (and that are not section headers) are silently
 *   skipped.
 * - Keys that appear before any @c [section] header are stored under the
 *   empty-string section key @c "".
 *
 */
#pragma once

#include <map>
#include <string>

namespace mesh_sim
{

/**
 * @brief Two-level string map produced by @ref parseIni.
 *
 * Indexed as @c ini[section][key]. The outer key is the section name
 * (e.g. @c "scenario", @c "channel"). Keys that appear before any
 * @c [section] header in the file are stored under the empty string @c "".
 */
using IniMap = std::map<std::string, std::map<std::string, std::string>>;


/**
 * @brief Parse an INI-style configuration file into a two-level string map.
 *
 * Reads @c path line by line, strips @c # and @c ; comments, trims
 * whitespace, skips blank lines, and populates an @ref IniMap.
 *
 * @param path  Filesystem path to the @c .ini file.
 * @return Fully populated @ref IniMap; empty if the file has no valid entries.
 * @throws std::runtime_error if the file cannot be opened.
 */
IniMap parseIni(const std::string& path);


/**
 * @brief Retrieve a string value from an @ref IniMap.
 *
 * Returns @p def when either the section or the key is absent, so callers
 * never need to check for missing entries explicitly.
 *
 * @param ini      Map produced by @ref parseIni.
 * @param section  Section name (e.g. @c "channel").
 * @param key      Key within the section (e.g. @c "frequency_ghz").
 * @param def      Default value returned when the entry is not found.
 *                 Defaults to an empty string.
 * @return The stored string value, or @p def if not found.
 */
std::string iniGet(const IniMap& ini,
                   const std::string& section,
                   const std::string& key,
                   const std::string& def = "");


/**
 * @brief Retrieve a boolean value from an @ref IniMap.
 *
 * Calls @ref iniGet and interprets the result case-insensitively:
 * - @c true  for @c "true", @c "1", or @c "yes".
 * - @c false for everything else (including @c "false", @c "0", @c "no").
 *
 * Returns @p def when the section or key is absent.
 *
 * @param ini      Map produced by @ref parseIni.
 * @param section  Section name.
 * @param key      Key within the section.
 * @param def      Default value returned when the entry is not found.
 * @return Parsed boolean value, or @p def if not found.
 */
bool iniGetBool(const IniMap& ini,
                const std::string& section,
                const std::string& key,
                bool def);

}  // namespace mesh_sim
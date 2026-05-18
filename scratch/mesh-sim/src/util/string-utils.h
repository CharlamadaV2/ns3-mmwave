/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

/** @brief
*/
namespace mesh_sim
{

/**
 * Trim leading and trailing whitespace from a string.
 */
 /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
std::string trimStr(const std::string& s);

/**
 * Split a string by tab characters and return the tokens.
 */
 /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
std::vector<std::string> splitTab(const std::string& line);

/**
 * Format a time_point as an ISO 8601 UTC string (e.g. "2026-03-27T14:30:00Z").
 */
 /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
std::string toIso8601(const std::chrono::system_clock::time_point& tp);

/**
 * Resolve a possibly-relative path against a base directory.
 * Returns the path unchanged if it is already absolute or empty.
 */
 /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
std::string resolvePath(const std::string& base_dir, const std::string& path);

/**
 * Return the parent directory of the given path.
 * Returns "." if the path contains no directory separator.
 */
 /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
std::string dirOf(const std::string& path);

/**
 * Parse a comma-separated string of seed values into a vector.
 * Skips empty tokens (e.g. "1,,3" -> {1, 3}).
 * Exits with an error message if any token is not a valid unsigned integer.
 */
 /**
 * Description of what the method does.
 *
 * @param input Description of parameter.
 * @return Description of return value.
 * @throws Exception Description of exception.
 */
std::vector<uint32_t> parseSeedList(const std::string& arg);

}  // namespace mesh_sim

/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file string-utils.h
 * @brief String and path manipulation utilities shared across all modules.
 *
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace mesh_sim
{

/**
 * @brief Strip leading and trailing ASCII whitespace from a string.
 *
 * Whitespace characters removed: space (@c ' '), horizontal tab (@c '\t'),
 * carriage return (@c '\r'), and newline (@c '\n').
 * Returns an empty string if @p s contains only whitespace.
 *
 * @param s  Input string.
 * @return Trimmed copy of @p s.
 */
std::string trimStr(const std::string& s);


/**
 * @brief Split a string on tab characters and return all tokens.
 *
 * Consecutive tab characters produce empty tokens between them (no collapsing).
 * An input with no tab characters returns a single-element vector containing
 * the whole string.
 *
 * @param line  Input string to split.
 * @return Vector of tokens in the order they appear in @p line.
 */
std::vector<std::string> splitTab(const std::string& line);


/**
 * @brief Format a @c system_clock::time_point as an ISO 8601 UTC string.
 *
 * Output format: @c "YYYY-MM-DDTHH:MM:SSZ" (e.g. @c "2026-03-27T14:30:00Z").
 * Sub-second precision is discarded. Conversion uses @c gmtime_r (POSIX);
 * this function is not portable to Windows without a compatibility shim.
 *
 * @param tp  Time point to format.
 * @return ISO 8601 UTC string.
 */
std::string toIso8601(const std::chrono::system_clock::time_point& tp);


/**
 * @brief Resolve a possibly-relative path against a base directory.
 *
 * Returns @p path unchanged if it is:
 * - empty, or
 * - already absolute (@c std::filesystem::path::is_absolute).
 *
 * Otherwise returns @c base_dir / @c path as an absolute string.
 * Used by @ref ConfigLoader to resolve @c nodes.json and @c buildings.json
 * relative to the scenario directory rather than the working directory.
 *
 * @param base_dir  Directory to use as the resolution root.
 * @param path      Path to resolve; may be absolute, relative, or empty.
 * @return Resolved path string.
 */
std::string resolvePath(const std::string& base_dir, const std::string& path);


/**
 * @brief Return the parent directory component of a file path.
 *
 * Delegates to @c std::filesystem::path::parent_path. Returns @c "." when
 * @p path contains no directory separator (i.e. it is a bare filename).
 *
 * @param path  Filesystem path (absolute or relative).
 * @return Parent directory string, or @c "." if none.
 */
std::string dirOf(const std::string& path);


/**
 * @brief Parse a comma-separated string of seed values into a @c uint32_t vector.
 *
 * Splits @p arg on @c ',' and converts each non-empty token with
 * @c std::stoul. Empty tokens (e.g. the gap in @c "1,,3") are silently
 * skipped so that trailing commas and double-commas are tolerated.
 *
 * @note On an invalid token (non-numeric or out of @c uint32_t range) this
 *       function prints an error to @c stderr and calls @c std::exit(1).
 *       It does not throw, matching the hard-exit convention used elsewhere
 *       in the CLI layer.
 *
 * @param arg  Comma-separated seed string (e.g. @c "1,2,3,4,5").
 * @return Vector of parsed seed values in the order they appear in @p arg.
 */
std::vector<uint32_t> parseSeedList(const std::string& arg);

}  // namespace mesh_sim
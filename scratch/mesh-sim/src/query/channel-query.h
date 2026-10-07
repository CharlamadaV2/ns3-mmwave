/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file channel-query.h
 * @brief @c --channel-query worker: scores candidate start layouts through the
 *        run's channel path, one forked child per layout.
 *
 * The wire contract (@c mesh_channel_query_v1), isolation guarantee, limits
 * and error semantics are documented in @ref src_query.
 */
#pragma once

#include "src/cli/cli-parser.h"
#include "src/domain/sim-config.h"

#include <istream>
#include <ostream>

namespace mesh_sim
{

/**
 * @fn RunChannelQuery
 * @brief Serve @c mesh_channel_query_v1 requests until @c shutdown or EOF.
 *
 * @param cfg   Validated config with CLI overrides and RL control applied.
 * @param args  Parsed CLI arguments; seeds must resolve to exactly one planning seed.
 * @param in    NDJSON request stream.
 * @param out   NDJSON response stream; the first line is the init message.
 * @return 0 on @c shutdown or EOF, 1 on a startup or output-stream failure.
 *
 * This process creates no ns-3 object or random variable and stays
 * single-threaded; every layout is built and evaluated in a forked child.
 */
int RunChannelQuery(const SimConfig& cfg, const CliArgs& args, std::istream& in, std::ostream& out);

}  // namespace mesh_sim

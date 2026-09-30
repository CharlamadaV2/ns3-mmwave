/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Minimal stub for ns3/log.h so ns-3-free sources (eval, routing, traffic)
 * compile in standalone tests. Logging macros expand to no-ops.
 */
#pragma once

#include <sstream>

#define NS_LOG_COMPONENT_DEFINE(name)
#define NS_LOG_DEBUG(msg)   do { if (false) { std::ostringstream _s; _s << msg; } } while (0)
#define NS_LOG_LOGIC(msg)   do { if (false) { std::ostringstream _s; _s << msg; } } while (0)
#define NS_LOG_INFO(msg)    do { if (false) { std::ostringstream _s; _s << msg; } } while (0)

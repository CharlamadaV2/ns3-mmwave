/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file ns3-log-stub.h
 * @brief Stub for ns3/log.h so eval and routing sources compile without ns-3.
 *
 * Only the logging macros used in the tested sources are defined; they do
 * nothing at run time.
 */
#pragma once

#include <sstream>

#define NS_LOG_COMPONENT_DEFINE(name)
#define NS_LOG_DEBUG(msg)   do { if (false) { std::ostringstream _s; _s << msg; } } while (0)
#define NS_LOG_LOGIC(msg)   do { if (false) { std::ostringstream _s; _s << msg; } } while (0)
#define NS_LOG_INFO(msg)    do { if (false) { std::ostringstream _s; _s << msg; } } while (0)

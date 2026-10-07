/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file jammer-spec.h
 * @brief Plain-data description of one jammer emitter (loaded from @c jammers.json).
 *
 * Pure POD with no ns-3 dependency. Parsed by @c ConfigLoader, checked by
 * @c ConfigValidator, given a mobility model by @c TopologyBuilder and
 * consumed by @ref mesh_sim::JammerModel. Held in @ref mesh_sim::SimConfig::jammers.
 */

#pragma once

#include "src/domain/node-spec.h"

#include <string>
#include <vector>


namespace mesh_sim
{


/**
 * @brief One active window of a jammer, in simulation seconds since scenario start.
 *
 * The window is half-open: the jammer is on when @c start <= t < @c end.
 * @c ConfigValidator requires @c start >= 0 and @c end > @c start.
 */
struct Interval
{
	double start;	///< Window start, seconds (inclusive).
	double end;	///< Window end, seconds (exclusive).

};

/**
 * @brief Parameters of one jammer emitter.
 *
 * Fields have no in-class defaults; @c ConfigLoader fills every one. Loader
 * defaults when a key is absent: enabled=false, type="constant", tx_power_dbm=25,
 * tx_array_gain_dbi=12, duty_cycle=1, max_range_m=0, beamwidth_deg=360,
 * azimuth_deg=0, zenith_deg=0. Validation accepts type in {constant, random},
 * duty_cycle in [0,1] and beamwidth_deg in (0,360].
 */
struct JammerSpec
{
	//Enabled: false drops the jammer in JammerModel::Configure (loader default: false)
	bool enabled; 

	//ID: Record of Jammer Node ID 
	std::string id;

	//Constant: Sends jamming signal for duration at certain interval
	//Random: Based on sim seed, sends jamming signal for duration per random interval
	//Type: string "constant" or "random" (anything else fails validation)
	std::string type;

	//Target Frequency (MHz): empty = all; 1 value = spot +/-2.5 MHz; 2+ values = [min,max] band
	std::vector<double> target_freq;

	//Power: Power of Jamming in dBm
	double tx_power_dbm;

	//Gain: Transmission Array Gain of antenna in dBi
	double tx_array_gain_dbi;

	//Time Intervals: active windows in sim seconds; empty = always on
	std::vector<Interval> intervals;

	// 0..1: random = chance of full-power transmission each second;
	// constant = multiplier on interference power (always transmitting).
	double duty_cycle;

	//Max Range (m): receivers farther than this are unaffected; 0 = no cutoff
	double max_range_m;
	
	//Beamwidth Angle (deg): full apex angle of the hard cone; >= 360 = omni (valid range (0,360])
	double beamwidth_deg;

	//Azimuth: Degree of where jammer is pointing horizontally [0 (or 360) = North, 90 = East, 180 = South, 270 = West]
	double azimuth_deg;

	//Zenith: Degree of where jammer is pointing vertically [0 = Up, 90 = horizontal, 180 = Down]
	double zenith_deg;

	//Waypoints (jammer motion; consumed by TopologyBuilder)
	std::vector<Waypoint> waypoints;

	//Initial position (ENU m) and optional motion parameters
	Position position;
	Velocity velocity;
	RandomWalkParams random_walk;

};

}

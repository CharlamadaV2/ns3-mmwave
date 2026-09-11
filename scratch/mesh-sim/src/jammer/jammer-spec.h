/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/* @brief Jammer Node Properties
 * */

#pragma once

#include "src/domain/node-spec.h"

#include <string>
#include <vector>


namespace mesh_sim
{


/* @brief Vector of start and end time for schedule jamming
 * */	
struct Interval
{
	double start;	//Start time in seconds
	double end;	//End time in seconds

};

struct JammerSpec
{
	//Enabled: Info on Jammer existance status
	bool enabled; 

	//ID: Record of Jammer Node ID 
	std::string id;

	//Constant: Sends jamming signal for duration at certain interval
	//Random: Based on sim seed, sends jamming signal for duration per random interval
	//Type: [0 = Constant, 1 = Random] (Note: Other types are for future implementation)
	std::string type;

	//Target Frequency: Array of Jamming Frequencies
	std::vector<double> target_freq_mhz;

	//Power: Power of Jamming in dBm
	double tx_power_dbm;

	//Gain: Transmission Array Gain of antenna in dBi
	double tx_array_gain_dbi;

	//Time Intervals: Set of [start, end] time interval(s) for jamming 
	std::vector<Interval> intervals;

	//Duty Cycle: Time jammer is transmitting 
	double duty_cycle;

	//Max Range: Max range for interrupted receivers
	double max_range_m;
	
	//Beamwidth Angle: Apex degree of jamming cone (min: 0.0, max: 180)
	double beamwidth_deg;

	//Azimuth: Degree of where jammer is pointing horizontally [0 (or 360) = North, 90 = East, 180 = South, 270 = West]
	double azimuth_deg;

	//Zenith: Degree of where jammer is pointing vertically [0 = Up, 180 = Down]
	double zenith_deg;

	//Waypoints
	std::vector<Waypoint> waypoints;

	//Movement
	Position position;
	Velocity velocity;
	RandomWalkParams random_walk;

};

}

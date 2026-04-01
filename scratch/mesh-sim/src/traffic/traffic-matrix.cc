/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/traffic/traffic-matrix.h"

#include "ns3/double.h"

#include <algorithm>
#include <stdexcept>

namespace mesh_sim
{

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TrafficMatrix::TrafficMatrix(const SimConfig& cfg)
    : m_trafficCfg(cfg.mesh.traffic),
      m_nodeSpecs(cfg.nodes),
      m_tickS(cfg.tick_s)
{
    m_uniformRng = ns3::CreateObject<ns3::UniformRandomVariable>();
    m_expRng     = ns3::CreateObject<ns3::ExponentialRandomVariable>();
}

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------

void
TrafficMatrix::Initialize(uint32_t numNodes, double currentTime)
{
    m_numNodes = numNodes;
    m_flows.clear();

    const auto& topo = m_trafficCfg.flow_topology;

    if (topo == "all_pairs")
    {
        InitAllPairs(numNodes, currentTime);
    }
    else if (topo == "random_pairs")
    {
        InitRandomPairs(numNodes, currentTime);
    }
    else if (topo == "gateway")
    {
        InitGateway(numNodes, currentTime);
    }
    else
    {
        throw std::runtime_error("[TrafficMatrix] Unknown flow_topology: " + topo);
    }
}

// ---------------------------------------------------------------------------
// Flow factory
// ---------------------------------------------------------------------------

Flow
TrafficMatrix::MakeFlow(uint32_t src, uint32_t dst, double currentTime) const
{
    Flow f;
    f.src          = src;
    f.dst          = dst;
    f.demand_mbps  = m_trafficCfg.demand_mbps;
    f.start_time_s = currentTime;
    f.active       = true;

    if (m_trafficCfg.holding_time_s > 0.0)
    {
        f.end_time_s = currentTime + m_trafficCfg.holding_time_s;
    }

    if (m_trafficCfg.model == "on_off")
    {
        f.in_on_phase = true;
        m_expRng->SetAttribute("Mean", ns3::DoubleValue(m_trafficCfg.on_time_s));
        f.phase_end_s = currentTime + m_expRng->GetValue();
    }

    return f;
}

// ---------------------------------------------------------------------------
// Topology initializers
// ---------------------------------------------------------------------------

void
TrafficMatrix::InitAllPairs(uint32_t numNodes, double currentTime)
{
    for (uint32_t i = 0; i < numNodes; ++i)
    {
        for (uint32_t j = i + 1; j < numNodes; ++j)
        {
            m_flows.push_back(MakeFlow(i, j, currentTime));
        }
    }
}

void
TrafficMatrix::InitRandomPairs(uint32_t numNodes, double currentTime)
{
    for (uint32_t k = 0; k < m_trafficCfg.random_pair_count; ++k)
    {
        uint32_t src = m_uniformRng->GetInteger(0, numNodes - 1);
        uint32_t dst = m_uniformRng->GetInteger(0, numNodes - 1);
        while (dst == src)
        {
            dst = m_uniformRng->GetInteger(0, numNodes - 1);
        }
        m_flows.push_back(MakeFlow(src, dst, currentTime));
    }
}

void
TrafficMatrix::InitGateway(uint32_t numNodes, double currentTime)
{
    // Gateway node is index 0 unless gateway_node_id is set.
    uint32_t gw = 0;
    if (!m_trafficCfg.gateway_node_id.empty())
    {
        // Try numeric index first; fall back to matching node ID string.
        try
        {
            gw = static_cast<uint32_t>(std::stoul(m_trafficCfg.gateway_node_id));
        }
        catch (const std::invalid_argument&)
        {
            bool found = false;
            for (uint32_t i = 0; i < m_nodeSpecs.size(); ++i)
            {
                if (m_nodeSpecs[i].id == m_trafficCfg.gateway_node_id)
                {
                    gw = i;
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                throw std::runtime_error(
                    "[TrafficMatrix] gateway_node_id '" +
                    m_trafficCfg.gateway_node_id + "' not found in node list");
            }
        }
    }

    for (uint32_t i = 0; i < numNodes; ++i)
    {
        if (i == gw)
        {
            continue;
        }
        m_flows.push_back(MakeFlow(i, gw, currentTime));
    }
}

// ---------------------------------------------------------------------------
// Tick
// ---------------------------------------------------------------------------

void
TrafficMatrix::Tick(double currentTime)
{
    // Expire flows that have reached their end time.
    for (auto& f : m_flows)
    {
        if (f.end_time_s > 0.0 && currentTime >= f.end_time_s)
        {
            f.active = false;
        }
    }

    if (m_trafficCfg.model == "on_off")
    {
        TickOnOff(currentTime);
    }
    else if (m_trafficCfg.model == "poisson")
    {
        TickPoisson(currentTime);
    }
    // "constant" model: nothing else to do.

    // Remove dead flows (inactive + expired).
    m_flows.erase(
        std::remove_if(m_flows.begin(), m_flows.end(),
                        [](const Flow& f) {
                            return !f.active && f.end_time_s > 0.0;
                        }),
        m_flows.end());
}

// ---------------------------------------------------------------------------
// On-off state machine
// ---------------------------------------------------------------------------

void
TrafficMatrix::TickOnOff(double currentTime)
{
    for (auto& f : m_flows)
    {
        if (!f.active)
        {
            continue;
        }

        while (f.phase_end_s <= currentTime)
        {
            f.in_on_phase = !f.in_on_phase;

            double mean = f.in_on_phase ? m_trafficCfg.on_time_s
                                        : m_trafficCfg.off_time_s;
            m_expRng->SetAttribute("Mean", ns3::DoubleValue(mean));
            f.phase_end_s += m_expRng->GetValue();
        }
    }
}

// ---------------------------------------------------------------------------
// Poisson arrivals
// ---------------------------------------------------------------------------

void
TrafficMatrix::TickPoisson(double currentTime)
{
    // Expected arrivals this tick = arrival_rate_hz * tick_s.
    // Draw actual count from Poisson (approximated via the standard method).
    double lambda = m_trafficCfg.arrival_rate_hz * m_tickS;
    double L = std::exp(-lambda);
    double p = 1.0;
    uint32_t arrivals = 0;

    do
    {
        ++arrivals;
        p *= m_uniformRng->GetValue();
    } while (p > L);
    --arrivals;

    for (uint32_t k = 0; k < arrivals; ++k)
    {
        uint32_t src = m_uniformRng->GetInteger(0, m_numNodes - 1);
        uint32_t dst = m_uniformRng->GetInteger(0, m_numNodes - 1);
        while (dst == src)
        {
            dst = m_uniformRng->GetInteger(0, m_numNodes - 1);
        }

        Flow f = MakeFlow(src, dst, currentTime);

        // Assign exponentially distributed holding time if configured.
        if (m_trafficCfg.holding_time_s > 0.0)
        {
            m_expRng->SetAttribute("Mean",
                                   ns3::DoubleValue(m_trafficCfg.holding_time_s));
            f.end_time_s = currentTime + m_expRng->GetValue();
        }

        m_flows.push_back(f);
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

const std::vector<Flow>&
TrafficMatrix::GetActiveFlows() const
{
    return m_flows;
}

double
TrafficMatrix::GetDemand(uint32_t src, uint32_t dst) const
{
    double total = 0.0;
    for (const auto& f : m_flows)
    {
        if (!f.active)
        {
            continue;
        }
        if (f.src == src && f.dst == dst)
        {
            // For on_off model, only count ON-phase flows.
            if (m_trafficCfg.model == "on_off" && !f.in_on_phase)
            {
                continue;
            }
            total += f.demand_mbps;
        }
    }
    return total;
}

}  // namespace mesh_sim

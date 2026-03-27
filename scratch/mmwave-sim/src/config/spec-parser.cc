/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/config/spec-parser.h"

namespace mmwave_sim
{

NodeSpec
parseNodeSpec(const nlohmann::json& j)
{
    NodeSpec n;
    n.id       = j.at("id").get<std::string>();
    n.role     = j.at("role").get<std::string>();
    n.mobility = j.value("mobility", "fixed");

    if (j.contains("position"))
    {
        const auto& p = j["position"];
        n.position.x  = p.value("x", 0.0);
        n.position.y  = p.value("y", 0.0);
        n.position.z  = p.value("z", 0.0);
    }

    if (j.contains("velocity"))
    {
        const auto& v = j["velocity"];
        n.velocity.vx = v.value("vx", 0.0);
        n.velocity.vy = v.value("vy", 0.0);
        n.velocity.vz = v.value("vz", 0.0);
    }

    if (j.contains("random_walk"))
    {
        const auto& rw = j["random_walk"];
        if (rw.contains("bounds"))
        {
            const auto& b   = rw["bounds"];
            n.random_walk.x_min = b.value("x_min", -100.0);
            n.random_walk.x_max = b.value("x_max",  100.0);
            n.random_walk.y_min = b.value("y_min", -100.0);
            n.random_walk.y_max = b.value("y_max",  100.0);
        }
        n.random_walk.speed_mps = rw.value("speed_mps", 1.5);
    }

    return n;
}

BuildingSpec
parseBuildingSpec(const nlohmann::json& j)
{
    BuildingSpec b;
    b.id = j.value("id", "");

    if (j.contains("bounds"))
    {
        const auto& bnd = j["bounds"];
        b.x_min = bnd.value("x_min", 0.0);
        b.x_max = bnd.value("x_max", 1.0);
        b.y_min = bnd.value("y_min", 0.0);
        b.y_max = bnd.value("y_max", 1.0);
        b.z_min = bnd.value("z_min", 0.0);
        b.z_max = bnd.value("z_max", 1.0);
    }

    b.type      = j.value("type",      "Residential");
    b.ext_walls = j.value("ext_walls", "ConcreteWithWindows");
    b.n_floors  = j.value("n_floors",  1);
    return b;
}

}  // namespace mmwave_sim

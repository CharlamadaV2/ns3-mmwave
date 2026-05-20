/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/util/ini-parser.h"
#include "src/util/string-utils.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace mesh_sim
{

IniMap
parseIni(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    IniMap sections;
    std::string section;
    std::string line;

    while (std::getline(f, line))
    {
        auto cpos = line.find('#');
        if (cpos != std::string::npos)
        {
            line = line.substr(0, cpos);
        }
        cpos = line.find(';');
        if (cpos != std::string::npos)
        {
            line = line.substr(0, cpos);
        }

        line = trimStr(line);
        if (line.empty())
        {
            continue;
        }

        if (line.front() == '[' && line.back() == ']')
        {
            section = trimStr(line.substr(1, line.size() - 2));
        }
        else
        {
            auto eq = line.find('=');
            if (eq == std::string::npos)
            {
                continue;
            }
            std::string key = trimStr(line.substr(0, eq));
            std::string val = trimStr(line.substr(eq + 1));
            sections[section][key] = val;
        }
    }
    return sections;
}

std::string
iniGet(const IniMap& ini,
       const std::string& section,
       const std::string& key,
       const std::string& def)
{
    auto sit = ini.find(section);
    if (sit == ini.end())
    {
        return def;
    }
    auto kit = sit->second.find(key);
    if (kit == sit->second.end())
    {
        return def;
    }
    return kit->second;
}

bool
iniGetBool(const IniMap& ini,
           const std::string& section,
           const std::string& key,
           bool def)
{
    std::string v = iniGet(ini, section, key, def ? "true" : "false");
    std::string lv = v;
    std::transform(lv.begin(), lv.end(), lv.begin(), ::tolower);
    return (lv == "true" || lv == "1" || lv == "yes");
}

}  // namespace mesh_sim

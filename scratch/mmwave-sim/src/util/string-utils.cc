/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
#include "src/util/string-utils.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace mmwave_sim
{

std::string
trimStr(const std::string& s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos)
    {
        return "";
    }
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::vector<std::string>
splitTab(const std::string& line)
{
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, '\t'))
    {
        tokens.push_back(tok);
    }
    return tokens;
}

std::string
toIso8601(const std::chrono::system_clock::time_point& tp)
{
    auto tt = std::chrono::system_clock::to_time_t(tp);
    std::tm tm{};
    gmtime_r(&tt, &tm);
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

std::string
resolvePath(const std::string& base_dir, const std::string& path)
{
    if (path.empty())
    {
        return path;
    }
    namespace fs = std::filesystem;
    fs::path p(path);
    if (p.is_absolute())
    {
        return path;
    }
    return (fs::path(base_dir) / p).string();
}

std::string
dirOf(const std::string& path)
{
    namespace fs = std::filesystem;
    fs::path parent = fs::path(path).parent_path();
    if (parent.empty())
    {
        return ".";
    }
    return parent.string();
}

}  // namespace mmwave_sim

/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Deterministic ns-3 RNG stubs for traffic-matrix unit tests.
 * UniformRandomVariable returns a cycling sequence so tests are repeatable.
 * ExponentialRandomVariable always returns exactly the configured mean.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace ns3
{

// --- Ptr / CreateObject ---

template <typename T>
class Ptr
{
  public:
    Ptr() = default;
    explicit Ptr(T* p) : m_ptr(p) {}
    T* operator->() const { return m_ptr; }
    T& operator*() const { return *m_ptr; }
    operator bool() const { return m_ptr != nullptr; }
  private:
    T* m_ptr = nullptr;
};

template <typename T>
Ptr<T> CreateObject()
{
    return Ptr<T>(new T());
}

// --- DoubleValue ---

class DoubleValue
{
  public:
    DoubleValue() = default;
    explicit DoubleValue(double v) : m_val(v) {}
    double Get() const { return m_val; }
  private:
    double m_val = 0.0;
};

// --- RandomVariableStream stubs ---

class RandomVariableStream
{
  public:
    virtual ~RandomVariableStream() = default;
    virtual void SetAttribute(const std::string&, const DoubleValue&) {}
};

class UniformRandomVariable : public RandomVariableStream
{
  public:
    // Returns values cycling through m_sequence, or 0.5 if empty.
    double GetValue()
    {
        if (m_sequence.empty()) { return 0.5; }
        double v = m_sequence[m_idx % m_sequence.size()];
        ++m_idx;
        return v;
    }

    uint32_t GetInteger(uint32_t min, uint32_t max)
    {
        if (m_intSequence.empty())
        {
            // Default: cycle through the range starting at min.
            uint32_t v = min + (m_intIdx % (max - min + 1));
            ++m_intIdx;
            return v;
        }
        uint32_t v = m_intSequence[m_intIdx % m_intSequence.size()];
        ++m_intIdx;
        return v;
    }

    // Test hooks: preload return sequences.
    std::vector<double>   m_sequence;
    std::vector<uint32_t> m_intSequence;
    size_t m_idx    = 0;
    size_t m_intIdx = 0;
};

class ExponentialRandomVariable : public RandomVariableStream
{
  public:
    void SetAttribute(const std::string&, const DoubleValue& v) override
    {
        m_mean = v.Get();
    }

    // Deterministic: always returns the mean.
    double GetValue() { return m_mean; }

  private:
    double m_mean = 1.0;
};

}  // namespace ns3

/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Minimal stub for ns3/random-variable-stream.h so that routing code
 * compiles in standalone tests. Only the types referenced (transitively)
 * by traffic-matrix.h are stubbed here.
 */
#pragma once

namespace ns3
{

template <typename T>
class Ptr
{
  public:
    Ptr() = default;
    Ptr(T* p) : m_ptr(p) {}
    T* operator->() const { return m_ptr; }
    operator bool() const { return m_ptr != nullptr; }
  private:
    T* m_ptr = nullptr;
};

class RandomVariableStream {};
class UniformRandomVariable : public RandomVariableStream {};
class ExponentialRandomVariable : public RandomVariableStream {};

}  // namespace ns3

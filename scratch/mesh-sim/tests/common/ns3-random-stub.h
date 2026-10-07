/* -*- Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/**
 * @file ns3-random-stub.h
 * @brief Stub for ns3/random-variable-stream.h so routing code compiles
 *        without ns-3.
 *
 * Declares empty Ptr and random-variable types, enough for the header
 * traffic-matrix.h that mesh-router.h includes. Not behavioral.
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

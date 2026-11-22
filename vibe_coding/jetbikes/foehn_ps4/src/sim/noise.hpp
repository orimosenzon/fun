// Deterministic noise helpers. Port of js/sim/noise.js (the parts the runtime sim needs).
// mulberry32 drives the wind field; it must match the JS bit-for-bit so runs replay the same.
#pragma once
#include <cstdint>

namespace fohn {

// JS: Math.imul -> low 32 bits of a 32-bit integer product.
struct Mulberry32 {
  uint32_t a;
  explicit Mulberry32(uint32_t seed) : a(seed) {}
  double operator()() {
    a = a + 0x6d2b79f5u;
    uint32_t t = (a ^ (a >> 15)) * (1u | a);
    t = (t + ((t ^ (t >> 7)) * (61u | t))) ^ t;
    return (double)((t ^ (t >> 14))) / 4294967296.0;
  }
};

} // namespace fohn

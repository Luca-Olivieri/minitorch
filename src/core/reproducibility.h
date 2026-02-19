#ifndef REPRODUCIBILITY_H
#define REPRODUCIBILITY_H

#include <random>
#include <sstream>
#include <string>

inline constexpr unsigned int seed = 42u;


inline std::mt19937 &prototype_rng() {
	static std::mt19937 proto(seed);
	return proto;
}

// Returns a new rng object (copy) each call, initialized with the
// current prototype state so callers receive independent RNGs
inline std::mt19937 get_rng() {
	return prototype_rng();
}

inline std::string serialize_rng_state() {
	std::ostringstream ss;
	ss << prototype_rng();
	return ss.str();
}

inline void deserialize_rng_state(const std::string &state) {
	std::istringstream ss(state);
	ss >> prototype_rng();
}

#endif

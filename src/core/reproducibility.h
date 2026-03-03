#ifndef REPRODUCIBILITY_H
#define REPRODUCIBILITY_H

#include <random>
#include <sstream>
#include <fstream>
#include <string>

#include "src/vendors/json.hpp"

using Json = nlohmann::json;

inline Json read_json(
	std::ifstream&& file
) {
    Json j;
    file >> j;
    return j;
}

const Json config = read_json(std::ifstream ("config/config.json"));
const unsigned int seed = config.value("seed", 42u);

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

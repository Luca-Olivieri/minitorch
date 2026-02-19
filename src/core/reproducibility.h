#ifndef REPRODUCIBILITY_H
#define REPRODUCIBILITY_H

#include "src/vendors/json.hpp"

#include <random>
#include <sstream>
#include <fstream>
#include <string>

using json = nlohmann::json;

inline json read_json(
    const std::string& path
) {
    std::ifstream file(path);
    json j;
    file >> j;
    return j;
}

const json config = read_json("/Users/lucaolivieri/Library/CloudStorage/GoogleDrive-luca.olivieri37@gmail.com/My Drive/coding/C++/minitorch/config/config.json");
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

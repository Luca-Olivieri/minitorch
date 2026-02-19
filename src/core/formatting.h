#ifndef FORMATTING_H
#define FORMATTING_H

#include <iostream>
#include <vector>
#include <map>

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector) {
    std::string out_string = "[";
    for (size_t i { 0 }; i < vector.size(); i++) {
        out_string += std::to_string(vector[i]);
        if (i != vector.size()-1) {
            out_string += ", ";
        }
    }
    out_string += "]";
    return os << out_string;
}

template <typename K, typename V>
inline std::ostream& operator<<(
        std::ostream& os,
        const std::map<K, V>& map
) {
    os << "{\n";
    for (auto it = map.cbegin(); it != map.cend(); ++it)
    {
        if (typeid(it->first) == typeid(std::string)) {
            os << "    \"" <<  it->first << "\"";
        }
        else {
            os << "    " <<  it->first;
        }
        os << ": " << it->second << ",\n";
    }
    os << "}\n";
    return os;
}


#endif

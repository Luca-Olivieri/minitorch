#ifndef FORMATTING_H
#define FORMATTING_H

#include <iostream>
#include <vector>
#include <map>

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector) {
    os << "[";
    for (size_t i { 0 }; i < vector.size(); i++) {
        os << vector[i];
        if (i != vector.size()-1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

template <typename K, typename V>
inline std::ostream& operator<<(std::ostream& os, const std::pair<K, V>& p) {
    if (typeid(p.first) == typeid(std::string)) {
            os << "    \"" <<  p.first << "\"";
        }
        else {
            os << "    " <<  p.first;
        }
    os << ": " << p.second;
    return os;
}

template <typename K, typename V>
inline std::ostream& operator<<(
        std::ostream& os,
        const std::map<K, V>& map
) {
    os << "{\n";
    for (auto it = map.cbegin(); it != map.cend(); ++it)
    {
        os << *it << ",\n";
    }
    os << "}\n";
    return os;
}


#endif

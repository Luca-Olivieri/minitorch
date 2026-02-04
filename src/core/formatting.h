#ifndef FORMATTING_H
#define FORMATTING_H

#include <iostream>
#include <vector>

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vector){
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

#endif

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <concepts>
#include <type_traits> 

// 1. Forward declaration of str
template <typename T> std::string str(const T& t);

// 2. Concept: Does the class have a const .to_string() method?
template<typename T>
concept StringableClass = requires(const T& t) {
    { t.to_string() } -> std::convertible_to<std::string>;
};

// --- IMPLEMENTATIONS ---

// Case A: Arithmetic types (int, float, etc.)
template <typename T>
requires std::is_arithmetic_v<T>
std::string str(const T& t) {
    return std::to_string(t);
}

// Case B: std::string (to prevent infinite recursion)
inline std::string str(const std::string& t) {
    return "\"" + t + "\""; 
}

// Case C: C-strings
inline std::string str(const char* t) {
    return std::string(t);
}

// Case D: Classes with .to_string()
template<StringableClass T>
std::string str(const T& obj) {
    return obj.to_string();
}

// Case E: Vectors (Recursive)
template <typename T>
std::string str(const std::vector<T>& vec) {
    std::string out = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        out += str(vec[i]); 
        if (i != vec.size() - 1) {
            out += ", ";
        }
    }
    out += "]";
    return out;
}

// --- GENERIC OPERATOR<< ---

// Concept: Can we call str(t)?
template<typename T>
concept Stringable = requires(const T& a) {
    { str(a) } -> std::convertible_to<std::string>;
};

// The Operator
// We exclude fundamental types (int, float) and std::string/char* 
// to avoid ambiguity with the Standard Library's own operators.
template<Stringable T>
requires (!std::is_fundamental_v<T> && 
          !std::same_as<T, std::string> && 
          !std::same_as<T, const char*> &&
          !std::same_as<T, char*>)
std::ostream& operator<<(std::ostream& os, const T& obj) {
    return os << str(obj);
}

#endif // UTILS_H

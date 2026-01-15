#include <iostream>
#include "ops.h" // Include your header

// --- Variable Implementation ---
Variable::Variable(float x) {
    value_ = x;
}
void Variable::forward() {
    std::cout << "forward";
}

// --- Addition Implementation ---

// Note: Do NOT use 'override' or 'virtual' in the .cpp file
void Addition::forward() {}

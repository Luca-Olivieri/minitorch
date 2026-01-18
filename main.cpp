#include <iostream>

#include "core/ops.h"
#include "core/tensors.h"

int do_nothing(
    int x,
    int y
){
    std::cout << x + y << '\n';
    return 1;
}

int main()
{   
    Tensor t({2, 3, 4, 5, 6});
    // Set some values manually since there are no setters yet
    t.m_flat_data[0] = 1.0f;
    t.m_flat_data[1] = 2.0f;
    t.m_flat_data[2] = 3.0f;
    t.m_flat_data[3] = 4.0f;
    t.m_flat_data[4] = 5.0f;
    t.m_flat_data[5] = 6.0f;

    std::cout << t << std::endl;

    return 0;
}

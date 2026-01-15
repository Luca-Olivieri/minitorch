#include <iostream>

#include "core/ops.h"

int do_nothing(
    int x,
    int y
){
    std::cout << x + y << '\n';
    return 1;
}

int main()
{   
    do_nothing(6, 7);

    Variable a(5.0f);

    a.forward();
    
    return EXIT_SUCCESS;
}

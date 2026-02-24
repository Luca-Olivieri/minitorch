#ifndef TEST_EXPAND_H
#define TEST_EXPAND_H

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_expand() {
    std::cout << "\n===[ test_expand.h ]===\n";

    // 1. Expand a leading singleton dimension and ensure storage is shared
    {
        Tensor t = Tensor::linspace({1, 2, 3}, 0.0f, 5.0f); // shape {1,2,3}

        // Keep the original storage shared_ptr
        auto orig_data_ptr = t.m_node->m_storage.m_flat_data;

        Tensor r = t.expand(0, 4); // shape -> {4,2,3}

        // The expanded storage must share the same underlying flat_data
        ASSERT_TRUE(r.m_node->m_storage.m_flat_data == orig_data_ptr, "expand shares underlying storage");

        // Mutating the view should affect the original (since it's a view)
        r[{3,1,2}] = 123.0f;
        ASSERT_EQ(t[{0,1,2}], 123.0f, "mutation through expanded view updates original");
    }
}

#endif

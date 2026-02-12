#ifndef TEST_SUM_H
#define TEST_SUM_H

#include "src/core/tensor_storages.h"
#include "tests/test_utils.h"

void test_storage_sum() {
    std::cout << "\n===[ test_sum.h ]===\n";

    // 1. 1D reduction -> scalar
    {
        TensorStorage a = TensorStorage::linspace({4}, 1.0f, 4.0f); // [1,2,3,4]
        TensorStorage s = TensorStorage::s_sum(a, 0);
        ASSERT_EQ(s.m_shape.size(), (size_t)0, "1D sum produces scalar shape");
        ASSERT_EQ(s.item(), 10.0f, "sum of 1..4 == 10");
    }

    // 2. 2D reduction along dim 0
    {
        TensorStorage a({2, 3});
        // fill rows: [1,2,3], [4,5,6]
        a.get_entry_ref(std::vector<size_t>{0,0}) = 1.0f;
        a.get_entry_ref(std::vector<size_t>{0,1}) = 2.0f;
        a.get_entry_ref(std::vector<size_t>{0,2}) = 3.0f;
        a.get_entry_ref(std::vector<size_t>{1,0}) = 4.0f;
        a.get_entry_ref(std::vector<size_t>{1,1}) = 5.0f;
        a.get_entry_ref(std::vector<size_t>{1,2}) = 6.0f;

        TensorStorage s0 = TensorStorage::s_sum(a, 0); // shape {3}
        ASSERT_EQ(s0.m_shape.size(), (size_t)1, "2D sum dim0 yields 1D shape");
        ASSERT_EQ(s0.get_entry_ref(std::vector<size_t>{0}), 5.0f, "sum col0 == 1+4");
        ASSERT_EQ(s0.get_entry_ref(std::vector<size_t>{1}), 7.0f, "sum col1 == 2+5");
        ASSERT_EQ(s0.get_entry_ref(std::vector<size_t>{2}), 9.0f, "sum col2 == 3+6");
    }

    // 3. 2D reduction along dim 1
    {
        TensorStorage a({2, 3});
        // fill rows: [1,2,3], [4,5,6]
        a.get_entry_ref(std::vector<size_t>{0,0}) = 1.0f;
        a.get_entry_ref(std::vector<size_t>{0,1}) = 2.0f;
        a.get_entry_ref(std::vector<size_t>{0,2}) = 3.0f;
        a.get_entry_ref(std::vector<size_t>{1,0}) = 4.0f;
        a.get_entry_ref(std::vector<size_t>{1,1}) = 5.0f;
        a.get_entry_ref(std::vector<size_t>{1,2}) = 6.0f;

        TensorStorage s1 = TensorStorage::s_sum(a, 1); // shape {2}
        ASSERT_EQ(s1.m_shape.size(), (size_t)1, "2D sum dim1 yields 1D shape");
        ASSERT_EQ(s1.get_entry_ref(std::vector<size_t>{0}), 6.0f, "sum row0 == 1+2+3");
        ASSERT_EQ(s1.get_entry_ref(std::vector<size_t>{1}), 15.0f, "sum row1 == 4+5+6");
    }

    // 4. Invalid dimension
    {
        TensorStorage a({2,2});
        ASSERT_THROWS(TensorStorage::s_sum(a, 2), std::invalid_argument);
    }
}

#endif

#ifndef TEST_MATMUL_H
#define TEST_MATMUL_H

#include "src/core/tensors.h"
#include "tests/test_utils.h"

void test_tensor_matmul() {
    std::cout << "\n===[ test_matmul.h ]===\n";

    // 1. 2x3 @ 3x2 -> 2x2
    {
        Tensor A({2, 3});
        A.fill_inplace(1.0f);
        // make A = [[1,2,3],[4,5,6]]
        A[{0,0}] = 1.0f; A[{0,1}] = 2.0f; A[{0,2}] = 3.0f;
        A[{1,0}] = 4.0f; A[{1,1}] = 5.0f; A[{1,2}] = 6.0f;

        Tensor B({3, 2});
        // B = [[7,8],[9,10],[11,12]]
        B[{0,0}] = 7.0f; B[{0,1}] = 8.0f;
        B[{1,0}] = 9.0f; B[{1,1}] = 10.0f;
        B[{2,0}] = 11.0f; B[{2,1}] = 12.0f;

        Tensor C = A.matmul(B);
        // Verify expansion pipeline values
        // Expected C = [[ 58,  64],[139,154]]
        ASSERT_EQ(C[{0,0}], 58.0f, "matmul 0,0");
        ASSERT_EQ(C[{0,1}], 64.0f, "matmul 0,1");
        ASSERT_EQ(C[{1,0}], 139.0f, "matmul 1,0");
        ASSERT_EQ(C[{1,1}], 154.0f, "matmul 1,1");
    }

    // 2. 1x3 @ 3x1 -> 1x1 (dot)
    {
        Tensor a({1,3});
        a[{0,0}] = 1.0f; a[{0,1}] = 2.0f; a[{0,2}] = 3.0f;
        Tensor b({3,1});
        b[{0,0}] = 4.0f; b[{1,0}] = 5.0f; b[{2,0}] = 6.0f;

        Tensor r = a.matmul(b);
        ASSERT_EQ(r[{0,0}], 32.0f, "1x3 @ 3x1 dot result");
    }

    // 3. Shape mismatch
    {
        Tensor x({2,2});
        Tensor y({3,3});
        ASSERT_THROWS(x.matmul(y), std::invalid_argument);
    }

    // 4. Backward gradients for 2x3 @ 3x2
    {
        Tensor A({2, 3});
        A.fill_inplace(0.0f);
        A[{0,0}] = 1.0f; A[{0,1}] = 2.0f; A[{0,2}] = 3.0f;
        A[{1,0}] = 4.0f; A[{1,1}] = 5.0f; A[{1,2}] = 6.0f;

        Tensor B({3, 2});
        B[{0,0}] = 7.0f; B[{0,1}] = 8.0f;
        B[{1,0}] = 9.0f; B[{1,1}] = 10.0f;
        B[{2,0}] = 11.0f; B[{2,1}] = 12.0f;

        Tensor C = A.matmul(B);
        C.backward();

        // Expected A grad: each row = sum over B rows -> [15,19,23]
        ASSERT_EQ(A.grad()[{0,0}], 15.0f, "A.grad 0,0");
        ASSERT_EQ(A.grad()[{0,1}], 19.0f, "A.grad 0,1");
        ASSERT_EQ(A.grad()[{0,2}], 23.0f, "A.grad 0,2");
        ASSERT_EQ(A.grad()[{1,0}], 15.0f, "A.grad 1,0");
        ASSERT_EQ(A.grad()[{1,1}], 19.0f, "A.grad 1,1");
        ASSERT_EQ(A.grad()[{1,2}], 23.0f, "A.grad 1,2");

        // Expected B grad: each row k summed over A rows -> [5,7,9] repeated across columns
        ASSERT_EQ(B.grad()[{0,0}], 5.0f, "B.grad 0,0");
        ASSERT_EQ(B.grad()[{0,1}], 5.0f, "B.grad 0,1");
        ASSERT_EQ(B.grad()[{1,0}], 7.0f, "B.grad 1,0");
        ASSERT_EQ(B.grad()[{1,1}], 7.0f, "B.grad 1,1");
        ASSERT_EQ(B.grad()[{2,0}], 9.0f, "B.grad 2,0");
        ASSERT_EQ(B.grad()[{2,1}], 9.0f, "B.grad 2,1");
    }

    // 5. Matrix [2,3] @ Vector [3] -> Vector [2]
    {
        Tensor A({2, 3});
        A.fill_inplace(0.0f);
        A[{0,0}] = 1.0f; A[{0,1}] = 2.0f; A[{0,2}] = 3.0f;
        A[{1,0}] = 4.0f; A[{1,1}] = 5.0f; A[{1,2}] = 6.0f;

        Tensor v({3});
        v[{0}] = 7.0f; v[{1}] = 9.0f; v[{2}] = 11.0f;

        Tensor r = A.matmul(v);
        ASSERT_EQ(r[{0}], 58.0f, "matmul matrix@vector 0");
        ASSERT_EQ(r[{1}], 139.0f, "matmul matrix@vector 1");

        r.backward();
        // A.grad rows should equal v
        ASSERT_EQ(A.grad()[{0,0}], 7.0f, "A.grad mv 0,0");
        ASSERT_EQ(A.grad()[{0,1}], 9.0f, "A.grad mv 0,1");
        ASSERT_EQ(A.grad()[{0,2}], 11.0f, "A.grad mv 0,2");
        ASSERT_EQ(A.grad()[{1,0}], 7.0f, "A.grad mv 1,0");
        ASSERT_EQ(A.grad()[{1,1}], 9.0f, "A.grad mv 1,1");
        ASSERT_EQ(A.grad()[{1,2}], 11.0f, "A.grad mv 1,2");

        // v.grad should be sum over rows of A -> [5,7,9]
        ASSERT_EQ(v.grad()[{0}], 5.0f, "v.grad mv 0");
        ASSERT_EQ(v.grad()[{1}], 7.0f, "v.grad mv 1");
        ASSERT_EQ(v.grad()[{2}], 9.0f, "v.grad mv 2");
    }

    // 6. Vector [3] @ Matrix [3,2] -> Vector [2]
    {
        Tensor u({3});
        u[{0}] = 1.0f; u[{1}] = 2.0f; u[{2}] = 3.0f;

        Tensor B({3,2});
        B[{0,0}] = 7.0f; B[{0,1}] = 8.0f;
        B[{1,0}] = 9.0f; B[{1,1}] = 10.0f;
        B[{2,0}] = 11.0f; B[{2,1}] = 12.0f;

        Tensor r = u.matmul(B);
        ASSERT_EQ(r[{0}], 58.0f, "matmul vector@matrix 0");
        ASSERT_EQ(r[{1}], 64.0f, "matmul vector@matrix 1");

        r.backward();

        // u.grad should be column-wise sums of B -> [15,19,23]
        ASSERT_EQ(u.grad()[{0}], 15.0f, "u.grad vm 0");
        ASSERT_EQ(u.grad()[{1}], 19.0f, "u.grad vm 1");
        ASSERT_EQ(u.grad()[{2}], 23.0f, "u.grad vm 2");

        // B.grad rows should equal u repeated across columns
        ASSERT_EQ(B.grad()[{0,0}], 1.0f, "B.grad vm 0,0");
        ASSERT_EQ(B.grad()[{0,1}], 1.0f, "B.grad vm 0,1");
        ASSERT_EQ(B.grad()[{1,0}], 2.0f, "B.grad vm 1,0");
        ASSERT_EQ(B.grad()[{1,1}], 2.0f, "B.grad vm 1,1");
        ASSERT_EQ(B.grad()[{2,0}], 3.0f, "B.grad vm 2,0");
        ASSERT_EQ(B.grad()[{2,1}], 3.0f, "B.grad vm 2,1");
    }

    // 7. Vector [3] @ Vector [3] -> scalar (dot)
    {
        Tensor p({3});
        p[{0}] = 1.0f; p[{1}] = 2.0f; p[{2}] = 3.0f;
        Tensor q({3});
        q[{0}] = 4.0f; q[{1}] = 5.0f; q[{2}] = 6.0f;

        Tensor s = p.matmul(q);
        ASSERT_EQ(s.item(), 32.0f, "vector·vector dot");

        s.backward();
        ASSERT_EQ(p.grad()[{0}], 4.0f, "p.grad dot 0");
        ASSERT_EQ(p.grad()[{1}], 5.0f, "p.grad dot 1");
        ASSERT_EQ(p.grad()[{2}], 6.0f, "p.grad dot 2");
        ASSERT_EQ(q.grad()[{0}], 1.0f, "q.grad dot 0");
        ASSERT_EQ(q.grad()[{1}], 2.0f, "q.grad dot 1");
        ASSERT_EQ(q.grad()[{2}], 3.0f, "q.grad dot 2");
    }
}

#endif

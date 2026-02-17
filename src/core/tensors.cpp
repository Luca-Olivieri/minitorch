#include <queue>
#include <map>
#include <set>
#include <algorithm>

#include "tensors.h"
#include "tensor_nodes.h"
#include "backward_ops.h"
#include "tensor_storages.h"

Tensor::Tensor(
        const std::vector<size_t> shape,
        const float value,
        const bool requires_grad
) {
    m_node = std::make_shared<TensorNode>(
        shape,
        value,
        requires_grad
    );
}

Tensor::Tensor(
        const std::shared_ptr<TensorNode> node
): m_node{node} {}

std::ostream& operator<<(
        std::ostream& os,
        const Tensor& tensor
) {
    os << tensor.m_node->m_storage;
    if (tensor.m_node->m_grad_fn) {
        os << " " << *tensor.m_node->m_grad_fn;
    }
    return os;
}
float& Tensor::operator[](
        const std::vector<size_t>& md_index
) const {
    if (m_node->m_storage.m_shape.empty()) { // Scalar case
        throw std::invalid_argument(std::format("\nScalar tensor cannot be access by index. Got index {}", md_index));
    }
    return m_node->m_storage.get_entry_ref((md_index));
}

float& Tensor::item() const {
    return m_node->m_storage.item();
} 

void Tensor::fill_inplace(
        const float value
) {
    m_node->m_storage.fill_inplace(value);
}

Tensor Tensor::fill(
        const float value
) const {
    TensorStorage out = m_node->m_storage.fill(value);
    std::shared_ptr<TensorNode> out_node = std::make_shared<TensorNode>(
        std::move(out)
    );
    return Tensor(out_node);
}

void Tensor::linspace_inplace(
        const float start,
        const float end
) {
    m_node->m_storage.linspace_inplace(start, end);
}

Tensor Tensor::linspace(
        const float start,
        const float end
) const {
    TensorStorage out = m_node->m_storage.linspace(start, end);
    std::shared_ptr<TensorNode> out_node = std::make_shared<TensorNode>(
        std::move(out)
    );
    return Tensor(out_node);
}

Tensor Tensor::linspace(
        const std::vector<size_t>& shape,
        const float start,
        const float end,
        const bool requires_grad
) {
    TensorStorage out = TensorStorage::linspace(std::move(shape), start, end);
    std::shared_ptr<TensorNode> out_node = std::make_shared<TensorNode>(
        std::move(out),
        requires_grad
    );
    return Tensor(out_node);
}

bool Tensor::is_contiguous() const {
    return m_node->m_storage.is_contiguous();
}

void Tensor::reset_grad() {
    // Replaces the gradient with a new zeroed tensor.
    // This ensures that any existing references to the old gradient (e.g. for higher order derivatives) are preserved intact.
    m_node->m_grad = std::make_shared<Tensor>(m_node->m_storage.m_shape);
    m_node->m_grad->fill_inplace(0.0f);
}

void Tensor::zero_grad() {
    if (m_node->m_grad) {
        reset_grad();
    }
    if (m_node->m_grad_fn) {
        m_node->m_grad_fn->reset_all_grads();
    }
}

void Tensor::detach_inplace() {
    m_node->m_grad_fn = nullptr;
}

bool Tensor::compute_requires_grad_from_operands(
    const std::vector<Tensor>& others
) const {
    std::vector<Tensor> operands = others;
    operands.push_back(*this); // add current instance to operands
    const bool requires_grad_or = std::any_of(
        operands.begin(),
        operands.end(),
        [](Tensor t){ return t.m_node->m_requires_grad; }
    );
    return requires_grad_or;
}

void Tensor::accumulate_grad(
        const Tensor& gradient
) {
    if (!m_node->m_grad) {
        // Initialize with zeros
        m_node->m_grad = std::make_shared<Tensor>(m_node->m_storage.m_shape);
        m_node->m_grad->fill_inplace(0.0f);
    }

    const Tensor current_grad(*m_node->m_grad);
    Tensor new_grad = current_grad + gradient;

    m_node->m_grad->m_node = new_grad.m_node;    
}

Tensor Tensor::operator+(
        const Tensor& other
) const {
    return apply_op_ag<TensorStorage::s_add, BackwardAdd>(other);
}

Tensor Tensor::operator/(
        const Tensor& other
) const {
    return apply_op_ag<TensorStorage::s_div, BackwardDiv>(other);
}

void Tensor::operator+=(
        const Tensor& other
) {
    TensorStorage::s_add_inplace(m_node->m_storage, other.m_node->m_storage);
}

Tensor Tensor::operator-() const {
    return apply_op_ag<TensorStorage::s_minus, BackwardMinus>();
}

Tensor Tensor::operator-(
        const Tensor& other
) const {
    return apply_op_ag<TensorStorage::s_sub, BackwardSub>(other);
}

Tensor Tensor::operator*(
        const Tensor& other
) const {
    return apply_op_ag<TensorStorage::s_mult, BackwardMult>(other);
}

Tensor Tensor::pow(
        const Tensor& other
) const {
    return apply_op_ag<TensorStorage::s_pow, BackwardPow>(other);
}

Tensor Tensor::log() const {
    return apply_op_ag<TensorStorage::s_log, BackwardLog>();
}

Tensor Tensor::sum(
        const size_t dim
) const {
    TensorStorage out_storage = TensorStorage::s_sum(
        m_node->m_storage, 
        dim
    );
    std::shared_ptr<TensorNode> out = std::make_shared<TensorNode>(
        std::move(out_storage)
    );
    out->m_grad_fn = std::make_unique<BackwardSum>(
        *this,
        dim,
        this->m_node->m_storage.m_shape[dim]
    );

    return Tensor(out);
}

Tensor Tensor::unsqueeze(
        const size_t dim
) const {
    TensorStorage out_storage = TensorStorage::s_unsqueeze(
        m_node->m_storage, 
        dim
    );
    std::shared_ptr<TensorNode> out = std::make_shared<TensorNode>(
        std::move(out_storage)
    );

    out->m_grad_fn = std::make_unique<BackwardUnsqueeze>(
        *this,
        dim
    );

    return Tensor(out);
}

Tensor Tensor::squeeze(
        const size_t dim
) const {
    TensorStorage out_storage = TensorStorage::s_squeeze(
        m_node->m_storage, 
        dim
    );
    std::shared_ptr<TensorNode> out = std::make_shared<TensorNode>(
        std::move(out_storage)
    );

    out->m_grad_fn = std::make_unique<BackwardSqueeze>(
        *this,
        dim
    );

    return Tensor(out);
}

Tensor Tensor::repeat(
        const size_t dim,
        const size_t times
) const {
    TensorStorage out_storage = TensorStorage::s_repeat(
        m_node->m_storage, 
        dim,
        times
    );
    std::shared_ptr<TensorNode> out = std::make_shared<TensorNode>(
        std::move(out_storage)
    );

    out->m_grad_fn = std::make_unique<BackwardRepeat>(
        *this,
        dim
    );

    return Tensor(out);
}

Tensor Tensor::clone() const {
    TensorStorage out_storage = m_node->m_storage.clone();
    std::shared_ptr<TensorNode> out = std::make_shared<TensorNode>(
        std::move(out_storage)
    );

    out->m_grad_fn = std::make_unique<BackwardClone>(*this);

    return Tensor(out);
}

Tensor Tensor::matmul(
        const Tensor& other
) const {
    const std::vector<size_t>& a_shape = this->m_node->m_storage.m_shape;
    const std::vector<size_t>& b_shape = other.m_node->m_storage.m_shape;

    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::invalid_argument(std::format("matmul requires 2D tensors, got {}D and {}D", a_shape.size(), b_shape.size()));
    }

    const size_t m = a_shape[0];
    const size_t k = a_shape[1];
    const size_t kb = b_shape[0];
    const size_t n = b_shape[1];

    if (k != kb) {
        throw std::invalid_argument(std::format("matmul inner dimensions must match ({} != {})", k, kb));
    }

    // Use unsqueeze->repeat->mult->sum pipeline
    const Tensor a_expanded = this->unsqueeze(2).repeat(2, n); // [m,k,1] -> [m,k,n]
    const Tensor b_expanded = other.unsqueeze(0).repeat(0, m); // [1,k,n] -> [m,k,n]

    const Tensor prod = a_expanded * b_expanded; // elementwise [m,k,n]
    const Tensor out = prod.sum(1); // sum over k -> [m,n]

    return out;
}

Tensor Tensor::grad() const {
    return *m_node->m_grad;
}

const std::vector<size_t>& Tensor::shape() const {
    return m_node->m_storage.m_shape;
}

std::map<TensorNode*, int> Tensor::compute_in_degree() const {
    std::map<TensorNode*, int> in_degree;
    std::queue<TensorNode*> bfs_queue;
    std::set<TensorNode*> visited;

    TensorNode* start = m_node.get();
    bfs_queue.push(start);
    visited.insert(start);
    in_degree[start] = 0; 
    
    // 1. Calculate in-degrees via BFS
    while(!bfs_queue.empty()) {
        TensorNode* u = bfs_queue.front();
        bfs_queue.pop();
        
        if (u->m_grad_fn) {
            const std::vector<Tensor> operands = u->m_grad_fn->get_operands();
            for(auto& op : operands) {
                TensorNode* v = op.m_node.get();
                in_degree[v]++;
                if(visited.find(v) == visited.end()) {
                    visited.insert(v);
                    bfs_queue.push(v);
                }
            }
        }
    }

    return in_degree;
}

void Tensor::topological_backprop(
        std::map<TensorNode*, int>& in_degree,
        const bool retain_graph
) const {
    // Process in topological order
    // Use Tensor objects in the queue (they hold shared_ptr to the node)
    std::queue<Tensor> process_queue;
    process_queue.push(*this);

    while(!process_queue.empty()) {
        Tensor u_tensor = process_queue.front();
        process_queue.pop();
        TensorNode* u = u_tensor.m_node.get();

        if (u->m_grad_fn) {
            // Push accumulated gradient to children
            u->m_grad_fn->compute_operands_grad(
                u_tensor
            );

            std::vector<Tensor> operands = u->m_grad_fn->get_operands();
            for (auto& op : operands) {
                TensorNode* v = op.m_node.get();
                in_degree[v]--;
                if(in_degree[v] == 0) {
                    process_queue.push(op);
                }
            }
            if (!retain_graph) {
                u->m_grad_fn = nullptr;
            }
        }
    }
}

void Tensor::backward(
        const bool retain_graph
) {
    if (m_node->m_requires_grad) {
        m_node->m_grad = std::make_shared<Tensor>(m_node->m_storage.m_shape);
        m_node->m_grad->fill_inplace(1.0f);
        
        // map to store in-degrees (number of parents in the computation graph that use this node)
        std::map<TensorNode*, int> in_degree { compute_in_degree() };
        // topological sort to ensure correctness for DAGs (shared nodes)
        topological_backprop(in_degree, retain_graph);
    }
    else {
        throw std::invalid_argument(std::format("\nCannot call backward on tensor with requires_grad=False. Likely, the graph has no leaf nodes requiring gradients."));
    }
}

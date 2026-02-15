#include <queue>
#include <map>
#include <set>

#include "tensors.h"
#include "tensor_nodes.h"
#include "backward_ops.h"
#include "tensor_storages.h"

Tensor::Tensor(
    std::vector<size_t> shape,
    float value
) {
    m_node = std::make_shared<TensorNode>(
        shape,
        value
    );
}

Tensor::Tensor(
    std::shared_ptr<TensorNode> node
): m_node{node} {}

std::ostream& operator<<(
    std::ostream& os,
    const Tensor& tensor
) {
    os << tensor.m_node->m_storage;
    if (tensor.m_node->m_bw_op) {
        os << " " << *tensor.m_node->m_bw_op;
    }
    return os;
}
float& Tensor::operator[](
    const std::vector<size_t>& md_index
) {
    if (m_node->m_storage.m_shape.empty()) { // Scalar case
        throw std::invalid_argument(std::format("\nScalar tensor cannot be access by index. Got index {}", md_index));
    }
    return m_node->m_storage.get_entry_ref((md_index));
}

float& Tensor::item() {
    return m_node->m_storage.item();
}

void Tensor::fill_inplace(
    float value
) {
    m_node->m_storage.fill_inplace(value);
}

Tensor Tensor::fill(
    float value
) {
    TensorStorage out = m_node->m_storage.fill(value);
    std::shared_ptr<TensorNode> out_node = std::make_shared<TensorNode>(
        std::move(out)
    );
    return Tensor(out_node);
}

void Tensor::linspace_inplace(
    float start,
    float end
) {
    m_node->m_storage.linspace_inplace(start, end);
}

Tensor Tensor::linspace(
    float start,
    float end
) {
    TensorStorage out = m_node->m_storage.linspace(start, end);
    std::shared_ptr<TensorNode> out_node = std::make_shared<TensorNode>(
        std::move(out)
    );
    return Tensor(out_node);
}

Tensor Tensor::linspace(
    std::vector<size_t> shape,
    float start,
    float end
) {
    TensorStorage out = TensorStorage::linspace(std::move(shape), start, end);
    std::shared_ptr<TensorNode> out_node = std::make_shared<TensorNode>(
        std::move(out)
    );
    return Tensor(out_node);
}

bool Tensor::is_contiguous() {
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
    if (m_node->m_bw_op) {
        m_node->m_bw_op->reset_all_grads();
    }
}

void Tensor::accumulate_grad(const Tensor& gradient, bool create_graph) {
    if (!m_node->m_grad) {
        // Initialize with zeros
        m_node->m_grad = std::make_shared<Tensor>(m_node->m_storage.m_shape);
        m_node->m_grad->fill_inplace(0.0f);
    }

    Tensor current_grad(*m_node->m_grad);
    Tensor new_grad = current_grad + gradient;

    if (!create_graph) {
        // Detach
        if (new_grad.m_node->m_bw_op) {
            new_grad.m_node->m_bw_op = nullptr;
        }
    }

    m_node->m_grad->m_node = new_grad.m_node;    
}

Tensor Tensor::operator+(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_add, BackwardAdd>(other);
}

Tensor Tensor::operator/(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_div, BackwardDiv>(other);
}

void Tensor::operator+=(
    const Tensor& other
) {
    TensorStorage::s_add_inplace(m_node->m_storage, other.m_node->m_storage);
}

Tensor Tensor::operator-() {
    return apply_op_ag<TensorStorage::s_minus, BackwardMinus>();
}

Tensor Tensor::operator-(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_sub, BackwardSub>(other);
}

Tensor Tensor::operator*(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_mult, BackwardMult>(other);
}

Tensor Tensor::pow(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_pow, BackwardPow>(other);
}

Tensor Tensor::log() {
    return apply_op_ag<TensorStorage::s_log, BackwardLog>();
}

Tensor Tensor::sum(
    const size_t dim
) {
    TensorStorage out_storage = TensorStorage::s_sum(
        m_node->m_storage, 
        dim
    );
    std::shared_ptr<TensorNode> out = std::make_shared<TensorNode>(
        std::move(out_storage)
    );
    out->m_bw_op = std::make_unique<BackwardSum>(
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

    out->m_bw_op = std::make_unique<BackwardUnsqueeze>(
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

    out->m_bw_op = std::make_unique<BackwardSqueeze>(
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

    out->m_bw_op = std::make_unique<BackwardRepeat>(
        *this,
        dim
    );

    return Tensor(out);
}

Tensor Tensor::grad() const {
    return *m_node->m_grad;
}

const std::vector<size_t>& Tensor::shape() {
    return m_node->m_storage.m_shape;
}

std::map<TensorNode*, int> Tensor::compute_in_degree() {
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
        
        if (u->m_bw_op) {
            std::vector<Tensor> operands = u->m_bw_op->get_operands();
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
    bool create_graph
) {
    // 2. Process in topological order
    std::queue<TensorNode*> process_queue;
    process_queue.push(m_node.get());
    
    while(!process_queue.empty()) {
        TensorNode* u = process_queue.front();
        process_queue.pop();
        
        if (u->m_bw_op) {
            // Push accumulated gradient to children
            u->m_bw_op->compute_operands_grad(
                Tensor(u->shared_from_this()),
                create_graph
            );
            
            std::vector<Tensor> operands = u->m_bw_op->get_operands();
            for (auto& op : operands) {
                TensorNode* v = op.m_node.get();
                in_degree[v]--;
                if(in_degree[v] == 0) {
                    process_queue.push(v);
                }
            }
        }
    }
}

void Tensor::backward(bool create_graph) {
    m_node->m_grad = std::make_shared<Tensor>(m_node->m_storage.m_shape);
    m_node->m_grad->fill_inplace(1.0f);
    
    // Topological sort to ensure correctness for DAGs (shared nodes)
    // Map to store in-degrees (number of parents in the computation graph that use this node)
    
    std::map<TensorNode*, int> in_degree { compute_in_degree() };

    topological_backprop(in_degree, create_graph);
}

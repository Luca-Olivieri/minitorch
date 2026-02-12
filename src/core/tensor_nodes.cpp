#include "tensor_nodes.h"
#include "backward_ops.h"
#include <queue>
#include <map>
#include <set>

TensorNode::TensorNode(
    std::vector<size_t> shape,
    float value
):
    m_storage{ shape, value },
    m_bw_op{ nullptr },
    m_grad{ nullptr } {}

TensorNode::TensorNode(
    TensorStorage storage
):
    m_storage{ std::move(storage) },
    m_bw_op{ nullptr },
    m_grad{ nullptr } {}

TensorNode::~TensorNode() = default;

std::ostream& operator<<(std::ostream& os, const TensorNode& tensor){
    return os << tensor.m_storage;
}

float& TensorNode::operator[](const std::vector<size_t>& md_index) {
    if (m_storage.m_shape.empty()) { // Scalar case
        throw std::invalid_argument(std::format("\nScalar tensor cannot be access by index. Got index {}", md_index));
    }
    return m_storage.get_entry_ref((md_index));
}

float& TensorNode::item() {
    return m_storage.item();
}

void TensorNode::fill_inplace(
    float value
) {
    m_storage.fill_inplace(value);
}

Tensor TensorNode::fill(
    float value
) {
    TensorStorage out = m_storage.fill(value);
    std::shared_ptr<TensorNode> out_node = std::make_shared<TensorNode>(
        std::move(out)
    );
    return Tensor(out_node);
}

void TensorNode::linspace_inplace(
    float start,
    float end
) {
    m_storage.linspace_inplace(start, end);
}

Tensor TensorNode::linspace(
    float start,
    float end
) {
    TensorStorage out = m_storage.linspace(start, end);
    std::shared_ptr<TensorNode> out_node = std::make_shared<TensorNode>(
        std::move(out)
    );
    return Tensor(out_node);
}

Tensor TensorNode::linspace(
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

bool TensorNode::is_contiguous() {
    return m_storage.is_contiguous();
}

Tensor TensorNode::operator+(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_add, BackwardAdd>(other);
}

Tensor TensorNode::operator/(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_div, BackwardDiv>(other);
}

void TensorNode::operator+=(
    const Tensor& other
) {
    TensorStorage::s_add_inplace(m_storage, other.m_node->m_storage);
}

Tensor TensorNode::operator-() {
    return apply_op_ag<TensorStorage::s_minus, BackwardMinus>();
}

Tensor TensorNode::operator-(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_sub, BackwardSub>(other);
}

Tensor TensorNode::operator*(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_mult, BackwardMult>(other);
}

Tensor TensorNode::pow(
    const Tensor& other
) {
    return apply_op_ag<TensorStorage::s_pow, BackwardPow>(other);
}

Tensor TensorNode::log() {
    return apply_op_ag<TensorStorage::s_log, BackwardLog>();
}

Tensor TensorNode::sum(
    const Tensor& a,
    const size_t dim
) {
    TensorStorage out_storage = TensorStorage::s_sum(
        a.m_node->m_storage, 
        dim
    );
    std::shared_ptr<TensorNode> out = std::make_shared<TensorNode>(std::move(out_storage));
    out->m_bw_op = std::make_unique<BackwardSum>(
        a,
        dim
    );

    return Tensor(out);
}

void TensorNode::reset_grad() {
    // Replaces the gradient with a new zeroed tensor.
    // This ensures that any existing references to the old gradient (e.g. for higher order derivatives) are preserved intact.
    m_grad = std::make_shared<TensorNode>(m_storage.m_shape);
    m_grad->fill_inplace(0.0f);
}

void TensorNode::zero_grad() {
    if (m_grad) {
        reset_grad();
    }
    if (m_bw_op) {
        m_bw_op->reset_all_grads();
    }
}

void TensorNode::accumulate_grad(const Tensor& gradient, bool create_graph) {
    if (!m_grad) {
        // Initialize with zeros
        m_grad = std::make_shared<TensorNode>(m_storage.m_shape);
        m_grad->fill_inplace(0.0f);
    }

    Tensor current_grad(m_grad);
    Tensor new_grad = current_grad + gradient;

    if (!create_graph) {
        // Detach
        if (new_grad.m_node->m_bw_op) {
            new_grad.m_node->m_bw_op = nullptr;
        }
    }

    m_grad = new_grad.m_node;    
}

std::map<TensorNode*, int> TensorNode::compute_in_degree() {
    std::map<TensorNode*, int> in_degree;
    std::queue<TensorNode*> bfs_queue;
    std::set<TensorNode*> visited;

    bfs_queue.push(this);
    visited.insert(this);
    in_degree[this] = 0; 
    
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

void TensorNode::topological_backprop(
    std::map<TensorNode*, int>& in_degree,
    bool create_graph
) {
    // 2. Process in topological order
    // Queue now holds nodes with in-degree 0 (ready to process)
    std::queue<TensorNode*> process_queue;
    process_queue.push(this);
    
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
            for(auto& op : operands) {
                TensorNode* v = op.m_node.get();
                in_degree[v]--;
                if(in_degree[v] == 0) {
                    process_queue.push(v);
                }
            }
        }
    }
}

void TensorNode::backward(bool create_graph) {
    m_grad = std::make_shared<TensorNode>(m_storage.m_shape);
    m_grad->fill_inplace(1.0f);
    
    // Topological sort to ensure correctness for DAGs (shared nodes)
    // Map to store in-degrees (number of parents in the computation graph that use this node)
    
    std::map<TensorNode*, int> in_degree { compute_in_degree() };

    topological_backprop(in_degree, create_graph);
}

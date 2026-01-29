#ifndef OPS_H
#define OPS_H

#include <iostream>

class Node {
public:
    float m_value { 0.0f };
    float m_grad { 0.0f };

    Node();
    
    Node(
        float value
    );
    
    virtual void backprop();
    
    virtual void reset_grads();
    
    virtual ~Node() = default;
    
    friend std::ostream& operator<<(std::ostream& os, const Node& node);
};

class Operation: public Node {
    public:
    // Subclasses MUST implement the destructor
    virtual ~Operation() = default;
    
    virtual void forward() = 0;
    void backward();
};

class UnaryOperation : public Operation {
    public:
    // 2. USE POINTERS (Critical)
    // We cannot store abstract classes by value. 
    // We use pointers to point to other operations in the graph.
    Node& m_x;
    
    // 3. Constructor
    // We need a way to connect these pointers when creating the object.
    UnaryOperation(
        Node& x
    ): m_x(x) {}
    
    void reset_grads() override;
};

class BinaryOperation : public Operation {
    public:
    // 2. USE POINTERS (Critical)
    // We cannot store abstract classes by value. 
    // We use pointers to point to other operations in the graph.
    Node& m_x;
    Node& m_y;
    
    // 3. Constructor
    // We need a way to connect these pointers when creating the object.
    BinaryOperation(
        Node& x,
        Node& y
    ): m_x(x), m_y(y) {}
    
    void reset_grads() override;
};

class Addition : public BinaryOperation {
    public:
    Addition(
        Node& x,
        Node& y
    ): BinaryOperation(x, y) {}
    
    // 5. Override methods
    void forward() override;

private:
    void backprop() override;
};

class Multiplication : public BinaryOperation {
public:
    Multiplication(
        Node& x,
        Node& y
    ): BinaryOperation(x, y) {}
    
    // 5. Override methods
    void forward() override;
    
private:
    void backprop() override;
};

class Squaration : public UnaryOperation {
public:
    Squaration(
        Node& x
    ): UnaryOperation(x) {}
    
    // 5. Override methods
    void forward() override;
    
private:
    void backprop() override;
};

#endif

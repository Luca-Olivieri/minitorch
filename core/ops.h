#ifndef OPS_H // 1. Check if the token is NOT defined
#define OPS_H // 2. Define the token

class Operation {
    public:
    // 2. Initialize variables to avoid garbage values
    float value = 0.0f;
    float grad = 0.0f;
    
    // 3. Keep the virtual destructor (this is correct!)
    virtual ~Operation() = default; 
    
    // 4. Make these "Pure Virtual" (= 0)
    // This forces children classes to implement them 
    // and prevents creating an instance of this base class.
    virtual void forward() = 0;
};

// 1. Inherit publicly
class BinaryOperation : public Operation {
    public:
    // 2. USE POINTERS (Critical)
    // We cannot store abstract classes by value. 
    // We use pointers to point to other operations in the graph.
    Operation* x;
    Operation* y;
    
    // 3. Constructor
    // We need a way to connect these pointers when creating the object.
    BinaryOperation(
        Operation* input_x,
        Operation* input_y
    ): x(input_x), y(input_y) {}
    
    // Note: We do NOT implement forward/backprop here.
    // This keeps BinaryOperation abstract.
};

class Variable {
public:
    float value_ { 0.0f };
    float grad_ { 0.0f };

    Variable(
        float x
    );

    void forward();
};

class Addition : public BinaryOperation {
public:
    // 4. Constructor
    // Pass the inputs up to the parent (BinaryOperation)
    Addition(
        Operation* x,
        Operation* y
    ): BinaryOperation(x, y) {}

    // 5. Override methods
    void forward() override;
};

#endif

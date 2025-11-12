# QAOA Workshop - Quantum Approximate Optimization Algorithm

**Author:** Alejandro Mata Ali (ITCL)

A comprehensive workshop implementation of the Quantum Approximate Optimization Algorithm (QAOA) using Qiskit 2.x, focused on solving the Traveling Salesman Problem (TSP) and general QUBO (Quadratic Unconstrained Binary Optimization) problems.

## 🎯 Overview

This workshop demonstrates how to:
- Formulate optimization problems as QUBO matrices
- Convert QUBO problems to Ising Hamiltonians
- Implement QAOA circuits using Qiskit
- Solve the Traveling Salesman Problem using quantum computing
- Compare quantum solutions with classical brute-force methods
- Analyze optimization convergence and solution quality

## 📁 Project Structure

```
qiskit2_qaoa_workshop/
├── qaoa_notebook.ipynb          # Main workshop notebook
├── qaoa_functions.py            # Core QAOA implementation
├── quantum_circuit.py           # Quantum circuit construction and evaluation
├── auxiliary_functions.py       # Problem formulation and utilities
├── requirements.txt             # Python dependencies
├── tsp/                        # TSP-specific results
│   ├── cost_function_QUBO.pdf  # Optimization convergence plots
│   └── values_QUBO             # Saved optimization results
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Qiskit 2.x
- NumPy, SciPy, Matplotlib

### Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook qaoa_notebook.ipynb
```

## 📚 Core Components

### 1. QAOA Implementation (`qaoa_functions.py`)

- **`qubo_solver()`**: Main QAOA solver with multiple random initializations
- **`qaoa()`**: Wrapper function for optimization algorithms
- Supports both statevector and sampling-based execution
- Automatic convergence plotting and result saving

### 2. Quantum Circuits (`quantum_circuit.py`)

- **`qaoa_circuit_for_train()`**: Parameterized QAOA circuit construction
- **`qaoa_results()`**: Circuit execution and measurement
- **`evaluator()`**: Cost function evaluation for optimization
- Optimized circuit transpilation for different backends

### 3. Problem Formulation (`auxiliary_functions.py`)

#### TSP-Specific Functions:
- **`generate_problem()`**: Random TSP instance generation
- **`tsp_qubo_formulation()`**: TSP to QUBO conversion
- **`brute_force_tsp_solver()`**: Classical optimal solution
- **`binary_to_tsp_solution()`**: Binary to route conversion
- **`compare_solutions()`**: Quantum vs classical comparison

#### General QUBO Functions:
- **`cost_function()`**: QUBO cost evaluation
- **`qubo_to_ising()`**: QUBO to Ising Hamiltonian conversion
- **`brute_force_qubo_solver()`**: Exhaustive QUBO solver

## 🧮 Mathematical Background

### QAOA Algorithm

QAOA alternates between two Hamiltonians:
- **Problem Hamiltonian (H_P)**: Encodes the optimization objective
- **Mixer Hamiltonian (H_M)**: Provides quantum superposition

The QAOA ansatz is:
```
|ψ(β,γ)⟩ = e^(-iβ_p H_M) e^(-iγ_p H_P) ... e^(-iβ_1 H_M) e^(-iγ_1 H_P) |+⟩^⊗n
```

### TSP Formulation

The TSP is formulated as a QUBO problem with:
- **Variables**: x_{t,i} = 1 if city i is visited at time t
- **Objective**: Minimize total travel distance
- **Constraints**: 
  - Each time step visits exactly one city
  - Each city is visited exactly once

### QUBO to Ising Conversion

QUBO variables x ∈ {0,1} are mapped to Ising spins s ∈ {-1,+1} using:
```
s = 2x - 1
```

## 🔧 Usage Examples

### Basic QAOA Execution

```python
from qaoa_functions import qubo_solver
from auxiliary_functions import generate_problem, tsp_qubo_formulation

# Generate TSP instance
distance_matrix = generate_problem(n_cities=4, n_connections=4, distance_range=10)

# Convert to QUBO
qubo_matrix = tsp_qubo_formulation(distance_matrix, restriction_parameter=10)

# Solve with QAOA
result = qubo_solver(
    qubo_matrix=qubo_matrix,
    number_of_layers=2,
    number_of_initializations=5,
    number_of_shots=100000
)
```

### Custom Backend Configuration

```python
from qiskit_aer import AerSimulator

# GPU backend (if available)
gpu_backend = AerSimulator(device='GPU', max_parallel_shots=0)

# Statevector simulation
statevector_backend = AerSimulator(method='statevector')

result = qubo_solver(qubo_matrix, backend=gpu_backend)
```

## 📊 Results and Analysis

The workshop generates several outputs:

1. **Convergence Plots**: Cost function evolution during optimization
2. **Probability Analysis**: Success probability tracking
3. **Solution Comparison**: Quantum vs classical solution quality
4. **Approximation Ratios**: Performance metrics

### Key Metrics

- **Approximation Ratio**: `quantum_cost / optimal_cost`
- **Success Probability**: Probability of measuring optimal solution
- **Convergence Rate**: Optimization iterations to convergence

## ⚙️ Configuration Options

### QAOA Parameters

- **`number_of_layers`**: QAOA circuit depth (default: 2)
- **`number_of_initializations`**: Random restarts (default: 4)
- **`number_of_shots`**: Measurement samples (default: 100,000)
- **`optimizer`**: Classical optimizer ('COBYLA', 'BFGS', etc.)

### Problem Parameters

- **`restriction_parameter`**: Constraint penalty weight
- **`distance_range`**: TSP distance scale
- **`n_connections`**: Graph connectivity

## 🎓 Educational Objectives

This workshop teaches:

1. **Quantum Algorithm Design**: QAOA circuit construction
2. **Problem Formulation**: Classical to quantum problem mapping
3. **Hybrid Optimization**: Quantum-classical algorithm integration
4. **Performance Analysis**: Solution quality assessment
5. **Practical Implementation**: Real quantum hardware considerations

## 🔬 Advanced Features

### Statevector vs Sampling

- **Statevector**: Exact probability computation (simulator only)
- **Sampling**: Realistic quantum hardware simulation

### Multi-Backend Support

- CPU/GPU simulators
- Real quantum hardware (IBM Quantum, etc.)
- Custom backend configurations

### Optimization Tracking

- Real-time convergence monitoring
- Best solution probability tracking
- Automatic result visualization

## 📈 Performance Considerations

### Scalability

- Problem size limited by qubit count: n_cities → (n_cities-1)² qubits
- Circuit depth grows with QAOA layers
- Classical optimization overhead

### Optimization Tips

1. **Normalization**: Scale QUBO matrices to avoid numerical issues
2. **Initialization**: Multiple random starts improve solution quality
3. **Layer Depth**: Balance expressivity vs noise (typically 1-4 layers)
4. **Shot Count**: Higher shots improve expectation value accuracy

### Dependencies

If you encounter import errors, ensure all dependencies are installed:
```bash
pip install --upgrade qiskit qiskit-aer numpy scipy matplotlib
```

## 📖 References

1. Farhi, E., Goldstone, J., & Gutmann, S. (2014). A Quantum Approximate Optimization Algorithm. arXiv:1411.4028
2. Qiskit Documentation: https://qiskit.org/documentation/
3. QAOA Tutorial: https://qiskit.org/textbook/ch-applications/qaoa.html

## 🤝 Contributing

This is an educational workshop. Suggestions for improvements are welcome:

- Enhanced problem formulations
- Additional optimization algorithms
- Performance optimizations
- Documentation improvements

## 📄 License

This workshop is provided for educational purposes. Please cite appropriately if used in academic work.

---
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import json
from quantum_circuit import evaluator, qaoa_circuit_for_train
from auxiliary_functions import qubo_to_ising
from scipy.optimize import minimize


def qaoa(quantum_circuit:QuantumCircuit, qubo_matrix:np.ndarray, cost_values:list, best_probs:list,
         backend, number_of_shots:int, statevector:bool, best_solution:list):
    '''Helper function that wraps the evaluator function for optimization.
    
    Creates a closure around the evaluator function to fix all parameters except the angles,
    making it suitable for use with optimization algorithms.
    
    Args:
        quantum_circuit (QuantumCircuit): Parameterized quantum circuit for the QAOA
        qubo_matrix (np.ndarray): QUBO matrix defining the optimization problem
        cost_values (list): List to store cost function values during optimization
        number_of_shots (int, optional): Number of circuit executions. Defaults to 1024
        backend (Backend, optional): Quantum backend to execute circuits. Defaults to GPU AerSimulator
        
    Returns:
        callable: Function that takes angles as input and returns the expectation value
    '''
    def function(angles:np.ndarray):
        return evaluator(angles, quantum_circuit, qubo_matrix, cost_values, best_probs, backend, number_of_shots, statevector, best_solution)
    return function


def qubo_solver(qubo_matrix:np.ndarray, number_of_layers:int=2, number_of_initializations:int=4, number_of_shots:int=int(1e5),
                verbose:bool=True, backend=AerSimulator(device='GPU', max_parallel_shots=0, max_parallel_threads=0), optimizer:str='COBYLA',
                problem_name:str='QUBO', best_solution:list=None, best_real_cost:float=np.inf, folder:str=''):
    '''Solves a QUBO problem using QAOA (Quantum Approximate Optimization Algorithm).
    
    Args:
        qubo_matrix (np.ndarray): QUBO cost matrix defining the optimization problem
        number_of_layers (int, optional): Number of QAOA layers. Defaults to 2
        number_of_initializations (int, optional): Number of random initializations. Defaults to 4
        number_of_shots (int, optional): Number of measurement shots. Defaults to 100000
        verbose (bool, optional): Whether to print progress information. Defaults to True
        backend (Backend, optional): Quantum backend for circuit execution. Defaults to GPU AerSimulator
        optimizer (str, optional): Classical optimization method. Defaults to 'COBYLA'
        
    Returns:
        OptimizeResult: Optimization result object containing the best solution found
    '''
    TOLERANCE = 1e-3
    if type(backend) == AerSimulator:
        statevector = (backend.options['method'] == 'statevector')
    else:
        statevector = False
        
    if statevector:
        print('Using statevector method.')

    # Convert QUBO matrix to Ising Hamiltonian representation
    q_ising = qubo_to_ising(qubo_matrix)

    # Create parameterized quantum circuit for QAOA with specified number of layers
    quantum_circuit = qaoa_circuit_for_train(q_ising, number_of_layers, backend, statevector)

    # Initialize list to track cost values during optimization
    cost_values = []
    best_probs = []
    
    # Create objective function that will be minimized by the classical optimizer
    expectation_value_function = qaoa(quantum_circuit, qubo_matrix, cost_values, best_probs, backend, number_of_shots, statevector, best_solution)

    # Initialize variables to keep track of best solution found
    best_cost = np.inf
    best_result = 0
    
    # Perform multiple optimization runs with different random initial angles
    for i in range(number_of_initializations):
        if verbose: print('Try ', i+1, ' of ', number_of_initializations)
        
        # Initialize random angles between 0 and pi for QAOA parameters
        angles = np.random.rand(2*number_of_layers)*(np.pi)
        
        # Run classical optimization to find optimal QAOA parameters
        result = minimize(expectation_value_function, angles, method=optimizer, tol = TOLERANCE)
        
        # Update best result if current solution is better
        if result.fun < best_cost:
            best_cost   = result.fun
            best_result = result

    if statevector:
        problem_name += '_statevector'

    rho = min(cost_values)/best_real_cost
    # Plot convergence of the optimization process
    plt.figure('Cost function', figsize=(8,4))
    plt.plot(cost_values, marker='')
    if best_real_cost < np.inf:
        plt.plot(best_real_cost*np.ones(len(cost_values)), 'r', marker='', label=r'$\rho=$'+f'{rho:.2e}')
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Expectation value ⟨H⟩', fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{folder}/cost_function_{problem_name}.pdf')
    plt.show()

    if best_probs:
        plt.figure('Probability function', figsize=(8,4))
        plt.yscale('log')
        plt.plot(qubo_matrix.shape[0]/2**(qubo_matrix.shape[0])*np.ones(len(best_probs)), 'r', marker='')
        plt.plot(np.ones(len(best_probs)), 'r', marker='')
        plt.plot(best_probs, marker='', label=f'best prob={max(best_probs):.2e}')
        plt.xlabel('Iteration', fontsize=15)
        plt.ylabel('Probability of best', fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{folder}/best_probability_{problem_name}.pdf')
        plt.show()

    data = {
        'name': problem_name,
        'Costs': cost_values,
        'Probabilities': best_probs,
        'Best_angles': list(best_result.x),
        'Real_solution': best_solution,
        'Real_cost': best_real_cost
    }
    # Guardar el número en el archivo
    with open(f'{folder}/values_{problem_name}', "w") as file:
        json.dump(data, file)

    return best_result
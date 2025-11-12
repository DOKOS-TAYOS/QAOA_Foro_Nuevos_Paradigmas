import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ParameterVector
from auxiliary_functions import cost_function


def qaoa_circuit_for_train(ising_matrix:np.ndarray, number_of_layers:int, backend, statevector:bool) -> QuantumCircuit:
    '''Creates a QAOA (Quantum Approximate Optimization Algorithm) circuit for training.
    
    This function constructs a parameterized quantum circuit implementing the QAOA algorithm
    for solving optimization problems encoded in an Ising Hamiltonian. The circuit alternates
    between problem and mixer Hamiltonians for a specified number of layers.
    
    Args:
        ising_matrix (np.ndarray): Square matrix representing the Ising model Hamiltonian.
        number_of_layers (int): Number of QAOA layers/repetitions to apply.
        backend: Quantum backend to run the circuit on.
        statevector: If we want to work with the statevector.
    
    Returns:
        QuantumCircuit: Transpiled QAOA circuit ready for execution, including measurements or statevector.
        
    Note:
        The circuit is automatically transpiled to the specified backend with optimization
        level 2 before being returned.
    '''
    # Create parameter vector for the QAOA angles
    # We need 2 parameters per layer: beta (mixing angle) and gamma (problem Hamiltonian angle)
    angles = ParameterVector('angles', 2*number_of_layers)
    number_of_qubits = ising_matrix.shape[0]

    # Initialize quantum circuit with required number of qubits
    quantum_register = QuantumRegister(number_of_qubits, 'q')
    quantum_circuit  = QuantumCircuit(quantum_register, name='Ising')

    # Create initial superposition state by applying Hadamard gates to all qubits
    quantum_circuit.h(quantum_register)

    # Iterate through QAOA layers
    for l in range(number_of_layers):
        # Apply problem Hamiltonian evolution
        # First apply single-qubit rotations for diagonal terms
        for i in range(number_of_qubits):
            quantum_circuit.rz(angles[number_of_layers+l]*ising_matrix[i][i], quantum_register[i])
            # Then apply two-qubit interactions for off-diagonal terms
            for j in range(i+1, number_of_qubits):
                quantum_circuit.rzz(angles[number_of_layers+l]*ising_matrix[i][j], quantum_register[i], quantum_register[j])

        # Apply mixer Hamiltonian using RX rotations on all qubits
        for i in range(number_of_qubits):
            quantum_circuit.rx(angles[l], quantum_register[i])

    # Add measurement of all qubits at the end of circuit
    if statevector is False:
        quantum_circuit.measure_all()

    # Pre-transpile the circuit to the target backend for optimization
    quantum_circuit_transpiled = transpile(quantum_circuit, backend, optimization_level=2)

    return quantum_circuit_transpiled

def qaoa_results(ising_matrix:np.ndarray, number_of_layers:int, angles:list, backend, number_of_shots:int=1024, statevector:bool=False) -> dict:
    '''Creates and executes a QAOA circuit to solve an Ising optimization problem.
    
    Args:
        ising_matrix (np.ndarray): Square matrix representing the Ising model Hamiltonian.
        number_of_layers (int): Number of QAOA layers/repetitions to apply.
        angles (list): List of angles for the QAOA circuit parameters.
        backend: Quantum backend to execute the circuit on.
        number_of_shots (int): Number of measurements to perform.
        statevector: If we want to work with the statevector.

    Returns:
        dict: Dictionary containing the measurement counts from circuit execution.
              Keys are binary strings representing measured states,
              values are the number of times each state was measured.
    '''
    #--------------------------------------------------------------------------
    number_of_qubits = ising_matrix.shape[0]

    quantum_register = QuantumRegister(number_of_qubits, 'q')
    quantum_circuit = QuantumCircuit(quantum_register, name='Ising')

    quantum_circuit.h(quantum_register)

    for l in range(number_of_layers):
        for i in range(number_of_qubits):
            quantum_circuit.rz(angles[number_of_layers+l]*ising_matrix[i][i], quantum_register[i])
            for j in range(i+1, number_of_qubits):
                quantum_circuit.rzz(angles[number_of_layers+l]*ising_matrix[i][j], quantum_register[i], quantum_register[j])

        for i in range(number_of_qubits):# Mixer
            quantum_circuit.rx(angles[l], quantum_register[i])

    if statevector is False:
        quantum_circuit.measure_all()

    #Pretranspilamos esta parte ya
    quantum_circuit_transpiled = transpile(quantum_circuit, backend, optimization_level=2)
    if statevector:
        quantum_circuit_transpiled.save_statevector()
        result = backend.run(quantum_circuit_transpiled).result()#, shots=number_of_shots).result()
        statevector = result.get_statevector(quantum_circuit_transpiled)
        probs_dict = statevector.probabilities_dict()
        return probs_dict

    else:
        counts_dictionary  = backend.run(quantum_circuit_transpiled, shots=number_of_shots).result().get_counts()

        return counts_dictionary


def evaluator(angles:np.ndarray, quantum_circuit:QuantumCircuit, qubo_matrix:np.ndarray, cost_value:list,
              best_probs:list,  backend, number_of_shots:int, statevector:bool, best_solution:list) -> float:
    '''Evaluates a QAOA circuit and computes the expected value of the cost function.
    
    Executes a parameterized QAOA circuit and calculates the expected value of the cost function
    by sampling measurement outcomes and averaging the corresponding energies.
    
    Args:
        angles (np.ndarray): Array of angles for the QAOA variational parameters.
        quantum_circuit (QuantumCircuit): Parameterized quantum circuit for the QAOA.
        qubo_matrix (np.ndarray): QUBO matrix defining the optimization problem.
        cost_value (list): List to store cost function values during optimization.
        number_of_shots (int): Number of circuit executions/measurements.
        backend: Quantum backend to execute the circuit.
        
    Returns:
        float: Expected value of the cost function averaged over measurement outcomes.
    '''

    # Assign the variational parameters (angles) to the quantum circuit
    quantum_circuit_transpiled = quantum_circuit.assign_parameters(angles, inplace=False)
    # Transpile the circuit for the target backend with optimization level 2
    quantum_circuit_transpiled = transpile(quantum_circuit_transpiled, backend, optimization_level=2)
    if statevector:
        quantum_circuit_transpiled.save_statevector()

    # Execute the circuit and get the measurement counts
    if statevector:
        result = backend.run(quantum_circuit_transpiled, shots=number_of_shots).result()
        statevector = result.get_statevector(quantum_circuit_transpiled)
        probs_dict = statevector.probabilities_dict()
        expectation_value = 0

        # Calculate the weighted sum of the cost function values for each measured bitstring
        for string in probs_dict:
            expectation_value += cost_function(string[::-1], qubo_matrix)*probs_dict[string]

        # Store the average cost value for this iteration
        cost_value.append(expectation_value)
        if best_solution is not None:
            best_probs.append(0)
            for _best in best_solution:
                best_probs[-1] += probs_dict[_best[::-1]]

        # Return the expectation value normalized by the number of shots
        return expectation_value

    else:
        counts_dictionary = backend.run(quantum_circuit_transpiled, shots=number_of_shots).result().get_counts()
        # Plot histogram of measurement counts using Qiskit's visualization
        expectation_value = 0

        # Calculate the weighted sum of the cost function values for each measured bitstring
        for string in counts_dictionary:
            expectation_value += cost_function(string[::-1], qubo_matrix)*counts_dictionary[string]

        # Store the average cost value for this iteration
        cost_value.append(expectation_value/number_of_shots)
        if best_solution is not None:
            best_probs.append(0)
            for _best in best_solution:
                best_probs[-1] += counts_dictionary[_best[::-1]]/number_of_shots

        # Return the expectation value normalized by the number of shots
        return expectation_value/number_of_shots






import numpy as np
from itertools import permutations, product

def cost_function(qubo_vector:np.ndarray, qubo_matrix:np.ndarray) -> float: #string, matrix -> value
    '''Calculates the cost of a solution for a QUBO problem.

    This function implements the cost calculation according to the formula:
    C(qubo_vector) = qubo_vector^T qubo_matrix qubo_vector
    where qubo_vector is a binary vector and qubo_matrix is the cost matrix of the QUBO problem.

    Args:
        qubo_vector (np.ndarray): Binary vector representing a solution.
                       Elements must be 0 or 1. It can be an string of 0 and 1.
        qubo_matrix (np.ndarray): Square matrix that defines the weights of the QUBO problem.
                       Must have the same dimensions as qubo_vector.

    Returns:
        float: The total cost of the provided solution.

    Example:
        >>> qubo_vector = np.array([1, 0, 1])
        >>> qubo_matrix = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
        >>> cost_function(qubo_vector, qubo_matrix)
        5.0
    '''

    if isinstance(qubo_vector[0], str):
        qubo_vector = np.array([int(i) for i in qubo_vector])

    return qubo_vector @ qubo_matrix @ qubo_vector

def cost_function_ising(ising_vector:np.ndarray, qubo_matrix:np.ndarray) -> float: #string, matrix -> value
    '''Calculates the cost of a solution for an Ising problem, up to a multiplicative factor.

    This function implements the cost calculation for an Ising model according to the formula:
    C(ising_vector) = (2 + ising_vector)^T qubo_matrix ising_vector
    where ising_vector is a vector with values {-1,1} and qubo_matrix is the cost matrix of the QUBO problem.

    Args:
        ising_vector (np.ndarray): Vector representing a solution in Ising format.
                       Elements must be -1 or 1. It can be a string of 0 and 1 
                       that will be mapped to -1 and 1 respectively.
        qubo_matrix (np.ndarray): Square matrix that defines the weights of the QUBO problem.
                       Must have the same dimensions as ising_vector.

    Returns:
        float: The total cost of the provided solution in the Ising model.

    Example:
        >>> ising_vector = np.array([-1, 1, -1])
        >>> qubo_matrix = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
        >>> cost_function_Ising(ising_vector, qubo_matrix)
        -2.0
    '''

    if isinstance(ising_vector[0], str):
        ising_vector = np.array([ -1 if i=='0' else 1 for i in ising_vector ])

    return (2*np.ones(len(ising_vector)) + ising_vector) @ qubo_matrix @ ising_vector



def qubo_to_ising(qubo_matrix:np.ndarray)-> np.ndarray:
    '''Maps a QUBO (Quadratic Unconstrained Binary Optimization) matrix to its equivalent Ising model matrix.
    
    This function performs the transformation from QUBO to Ising model using the relation:
    s = 2x - 1, where x are QUBO variables (0,1) and s are Ising variables (-1,+1).
    
    Args:
        qubo_matrix (np.ndarray): Square matrix representing the QUBO problem Hamiltonian.
                            Must be symmetric.
    
    Returns:
        np.ndarray: Square matrix representing the equivalent Ising model Hamiltonian.
                   Has the same dimensions as the input matrix.
    
    Note:
        The transformation preserves the ground state energy and the relative energy gaps
        between states, only changing the encoding from binary to spin variables.
    '''

    dimension = qubo_matrix.shape[0]
    ising_matrix = np.zeros((dimension,dimension), dtype=float)

    for i in range(dimension):
        for j in range(dimension):
            ising_matrix[i][i] -= qubo_matrix[i][j]
        for j in range(i+1,dimension):
            ising_matrix[i][j] += qubo_matrix[i][j]

    return ising_matrix/2


# Linear counter
def idx(i:int, j:int, N:int)->int:
    return i * N + j

def generate_problem(n_cities: int, n_connections: int, distance_range: float) -> np.ndarray:
    """
    Generate a TSP problem instance.
    
    Args:
        n_cities: Number of cities in the problem.
        n_connections: Maximum number of outgoing connections per city.
        distance_range: Maximum distance between cities (distances will be in range [0, distance_range]).
    
    Returns:
        A numpy array of shape (n_cities, n_cities) representing the distance matrix,
        where distances[i, j] is the distance from city i to city j.
        Unreachable cities have a distance of infinity.
    """
    # Initialize distance matrix with infinity
    distances = np.ones((n_cities, n_cities), dtype=np.int32) * float('inf')
    
    for city in range(n_cities):
        # Get all other cities
        other_cities = np.arange(n_cities, dtype=np.int32)
        other_cities = other_cities[other_cities != city]
        
        # Randomly shuffle to select random connections
        other_cities = other_cities[np.random.permutation(len(other_cities))]
        
        # Connect to at most n_connections other cities
        for i in range(min(n_connections, len(other_cities))):
            destination = other_cities[i]
            # Generate random distance in [0, distance_range]
            distances[city, destination] = np.random.randint(0, int(distance_range))
    
    return distances

def binary_to_tsp_solution(binary_solution):
    """
    Translate binary solution from QAOA to TSP route representation.
    
    Args:
        binary_solution: Binary string or array representing the QAOA solution
    
    Returns:
        list: Route as a sequence of city indices, starting and ending at city 0
    """
    n_variables = int(np.sqrt(len(binary_solution)))  # Exclude the fixed starting city
    
    # Convert string to numpy array if needed
    if isinstance(binary_solution, str):
        binary_array = np.array([int(bit) for bit in binary_solution])
    else:
        binary_array = np.array(binary_solution)
    
    # Reshape to (time_step, city) matrix
    solution_matrix = binary_array.reshape(n_variables, n_variables)
    
    # Extract the route
    route = [] 
    
    for t in range(n_variables):
        # Find which city is visited at time step t
        city_visited = np.where(solution_matrix[t] == 1)[0]
        
        if len(city_visited) == 1:
            # Add 1 to account for the fact that city indices in matrix are 0-based
            # but represent cities 1 to n_variables (city 0 is fixed as start)
            route.append(city_visited[0])
        elif len(city_visited) == 0:
            # No city selected for this time step - constraint violation
            route.append(-1)  # Mark as invalid
        else:
            # Multiple cities selected - constraint violation
            route.append(-1)
            print('Repetition constraint:', solution_matrix)
    
    route.append(n_variables)  # Return to starting city
    
    return route

def tsp_cost(route, distance_matrix):
    """
    Calculate the total cost/distance of a TSP route.
    
    Args:
        route (list): List of city indices representing the route
        distance_matrix (np.ndarray): Distance matrix where distance_matrix[i][j] 
                                    is the distance from city i to city j
    
    Returns:
        float: Total distance of the route, or np.inf if route is invalid
    """
    
    total_distance = 0
    
    # Check each edge in the route
    for i in range(len(route)):
        current_city = route[i]
        previous_city = route[i-1]
        
        # Check for invalid city indices (marked as -1 for constraint violations)
        if current_city == -1 or previous_city == -1:
            return np.inf
        
        # Check if edge exists (not infinite distance)
        edge_distance = distance_matrix[previous_city, current_city]
        if edge_distance == np.inf or np.isnan(edge_distance):
            return np.inf
        
        total_distance += edge_distance
    
    return total_distance

def brute_force_tsp_solver(distance_matrix):
    """
    Solve TSP using brute force approach by trying all possible permutations.
    
    Args:
        distance_matrix: Distance matrix where distance_matrix[i][j] is the distance 
                        from city i to city j
    
    Returns:
        tuple: (best_route, min_distance) where best_route is a list of city indices
               representing the optimal tour, and min_distance is the total distance
    """
    
    n_cities = len(distance_matrix)
    
    # Generate all possible permutations of cities (excluding the starting city 0)
    cities = list(range(0, n_cities-1))
    min_distance = float('inf')
    best_route = None
    
    # Try all permutations
    for perm in permutations(cities):
        # Create complete route starting and ending at city n_cities
        route = list(perm) + [n_cities-1]
        
        # Calculate total distance for this route
        total_distance = tsp_cost(route, distance_matrix)
        
        # Update best solution if this route is better and valid
        if total_distance < min_distance:
            min_distance = total_distance
            best_route = route
    
    return best_route, min_distance

def tsp_qubo_formulation(distance_matrix: np.ndarray, restriction_parameter: float) -> np.ndarray:
    """
    Create QUBO formulation for the Traveling Salesman Problem.
    
    Args:
        distance_matrix: Square matrix of distances between cities
        restriction_parameter: Penalty parameter for constraint violations
        
    Returns:
        QUBO matrix representing the TSP problem
    """
    n_cities = len(distance_matrix)
    n_variables = n_cities - 1  # Excluding the starting city
    
    # Initialize QUBO matrix
    qubo_size = n_variables ** 2
    qubo_matrix = np.zeros((qubo_size, qubo_size), dtype=float)
    
    # Cost term: minimize travel distances
    for t in range(n_variables):
        for i in range(n_variables):
            start_index = idx(t, i, n_variables)
            
            if t == 0:
                # First step: from last city to city i
                qubo_matrix[start_index, start_index] += distance_matrix[n_variables, i]
                
                # Add distances to next cities
                for j in range(n_variables):
                    next_index = idx(t + 1, j, n_variables)
                    if i != j:
                        qubo_matrix[start_index, next_index] += distance_matrix[i, j]
                    else:
                        # Penalty for visiting same city consecutively
                        qubo_matrix[start_index, next_index] += restriction_parameter
                        
            elif t == n_variables - 1:
                # Last step: from city i back to starting city
                qubo_matrix[start_index, start_index] += distance_matrix[i, n_variables]
                
            else:
                # Intermediate steps
                for j in range(n_variables):
                    next_index = idx(t + 1, j, n_variables)
                    if i != j:
                        qubo_matrix[start_index, next_index] += distance_matrix[i, j]
                    else:
                        # Penalty for visiting same city consecutively
                        qubo_matrix[start_index, next_index] += restriction_parameter

    # Constraint 1: Each time step must visit exactly one city
    for t in range(n_variables):
        for i in range(n_variables):
            start_index = idx(t, i, n_variables)
            qubo_matrix[start_index, start_index] -= restriction_parameter
            
            for j in range(i + 1, n_variables):
                cross_index = idx(t, j, n_variables)
                qubo_matrix[start_index, cross_index] += 2 * restriction_parameter
    
    # Constraint 2: Each city must be visited exactly once
    for i in range(n_variables):
        for t in range(n_variables):
            start_index = idx(t, i, n_variables)
            qubo_matrix[start_index, start_index] -= restriction_parameter
            
            for t_prime in range(t + 1, n_variables):
                cross_index = idx(t_prime, i, n_variables)
                qubo_matrix[start_index, cross_index] += 2 * restriction_parameter
                
    return qubo_matrix

def brute_force_qubo_solver(qubo_matrix):
    """
    Solve QUBO problem using brute force approach by trying all possible binary combinations.
    
    Args:
        qubo_matrix (np.ndarray): Square QUBO matrix where qubo_matrix[i][j] represents
                                 the weight between variables i and j
    
    Returns:
        tuple: (best_solution, min_cost) where best_solution is a binary vector
               representing the optimal solution, and min_cost is the minimum cost value
    """
    
    n_variables = qubo_matrix.shape[0]
    min_cost = float('inf')
    best_solution = None
    
    # Try all possible binary combinations (2^n possibilities)
    # Generate all possible binary combinations using itertools
    for binary_combination in product([0, 1], repeat=n_variables):
        # Convert tuple to numpy array
        binary_solution = np.array(binary_combination)
        
        # Calculate cost for this solution using the cost function
        current_cost = cost_function(binary_solution, qubo_matrix)
        
        # Update best solution if this cost is better
        if current_cost < min_cost:
            min_cost = current_cost
            best_solution = binary_solution.copy()
    
    return best_solution, min_cost

def compare_solutions(distance_matrix, qubo_matrix, qaoa_solution):
    """
    Compare QAOA solution with brute force optimal solution.
    
    Args:
        distance_matrix: Distance matrix of the TSP problem
        qubo_matrix: QUBO matrix representation of the problem
        qaoa_solution: Binary solution from QAOA
    
    Returns:
        dict: Comparison results containing routes, distances, and quality metrics
    """
    # Get brute force solution
    print("=== BRUTE FORCE SOLUTION ===")
    bf_route, bf_distance = brute_force_tsp_solver(distance_matrix)
    
    print(f"Optimal route: {' -> '.join(map(str, bf_route))}")
    print(f"Optimal distance: {bf_distance}")
    
    print("\n=== QUBO SOLUTION ===")
    qubo_solution, qubo_cost = brute_force_qubo_solver(qubo_matrix)
    qubo_route = binary_to_tsp_solution(qubo_solution)
    qubo_real_cost = tsp_cost(qubo_route, distance_matrix)
    print(f"QUBO binary solution: {''.join(map(str, qubo_solution))}")
    print(f"QUBO route: {' -> '.join(map(str, qubo_route))}")
    print(f"QUBO distance: {qubo_real_cost}")
    print(f"QUBO cost: {qubo_cost}")
    
    if qubo_real_cost != np.inf and bf_distance > 0:
        qubo_approximation_ratio = qubo_real_cost / bf_distance
        print(f"QUBO approximation ratio: {qubo_approximation_ratio:.4f}")
        print(f"QUBO quality: {(1/qubo_approximation_ratio)*100:.2f}% of optimal")
    else:
        print("QUBO solution is invalid")
    
    results = {
        'brute_force_route': bf_route,
        'brute_force_distance': bf_distance,
        'qubo_route': qubo_route,
        'qubo_distance': qubo_real_cost,
        'qubo_cost': qubo_cost
    }
    
    if qaoa_solution is not None:
        print("\n=== QAOA SOLUTION ===")
        qaoa_route = binary_to_tsp_solution(qaoa_solution)
        qaoa_distance = tsp_cost(qaoa_route, distance_matrix)
        print(f"QAOA route: {' -> '.join(map(str, qaoa_route))}")
        print(f"QAOA distance: {qaoa_distance}")
        
        if qaoa_distance != np.inf and bf_distance > 0:
            approximation_ratio = qaoa_distance / bf_distance
            print(f"\nApproximation ratio: {approximation_ratio:.4f}")
            print(f"Quality: {(1/approximation_ratio)*100:.2f}% of optimal")
        else:
            print("\nQAOA solution is invalid")
        
        results.update({
            'qaoa_route': qaoa_route,
            'qaoa_distance': qaoa_distance,
            'approximation_ratio': qaoa_distance / bf_distance if qaoa_distance != np.inf and bf_distance > 0 else np.inf
        })
    
    return results

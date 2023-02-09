OPTIONS = {"B1", "B2", "G1"}


# Check if the current state is a valid soluion
def is_valid_state(state):
    # The current state is valid is there is a unique student in each seat
    return len(state) == 3


# Get list of potential next steps
def get_candidates(state):
    print(state)
    print(list(OPTIONS.difference(set(state))))
    return list(OPTIONS.difference(set(state)))


# Recursively, perform a depth-first search to find valid solutions
def search(state, solutions):
    # Check is the state is valid
    if is_valid_state(state):
        # Add a copy of the valid state to list of solutions
        solutions.append(state.copy())
        # return # uncomment if you only need to find one valid solution

    # Iterate through the candidates that can be used
    # to construct the next state
    for candidate in get_candidates(state):
        # Add candidate to the current state
        state.append(candidate)
        # Call search function with updated state
        search(state, solutions)
        # Remove the current candidate from the current state
        state.remove(candidate)


# Entry point to the program
# responsible for returning the valid solutions
def solve():
    solutions = []
    state = []
    search(state, solutions)
    return solutions


if __name__ == "__main__":
    solutions = solve()
    print(solutions)
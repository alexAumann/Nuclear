# using code from https://christianjmills.com/posts/backtracking-notes/index.html
import re
import numpy as np
from isotope import ion
from nuclear import Nuclear


class BackTrack(Nuclear):
    # The size of the cycle is the number of distinct isotopes present

    def __init__(self, data):
        Nuclear.__init__(self, data)  # is this right?
        self.nuc_change = [0, 0, 0]
        self.rates = np.delete(np.array(self.get_rates()[1:]), [3, 4], 1)
        # I'm not considering (n,p) reactions

    # DON'T WANT INTERACTION TO BE SELF-INTERSECTING
    # Check if the current state is a valid solutions
    @staticmethod
    def is_valid_state(cycle):
        # n + 1 is current element in list ?
        if len(cycle) == 1:
            return False
        else:
            if cycle[0] == cycle[-1]:
                # I want to check if the cycle is a cycle
                return True
            else:
                return False

    @ staticmethod
    def remove_val(lst, val):
        return [value for value in lst if value != val]

    def find_candidate(self, cycle, lst2, min_rate):

        try:
            ion(N=cycle[-1].N + lst2[0], Z=cycle[-1].Z + lst2[1], A=cycle[-1].A + lst2[2])
        except AssertionError:
            return None

        temp_cycle = cycle.copy()
        temp_cycle.append(ion(N=cycle[-1].N + lst2[0], Z=cycle[-1].Z + lst2[1], A=cycle[-1].A + lst2[2]))
        self.nuc_change = lst2
        i = self.check_priority()
        elem = self.rates[np.where(self.rates[:, 0] == ion(temp_cycle[-2 + i]))]

        if len(elem) > 0 and self.str_to_float(elem[0, self.reactions.index(self.nuc_change) + 1]) > min_rate:
            if temp_cycle[-1] in cycle[:-1]:
                return None
            else:
                return temp_cycle[-1]
        else:
            return None

    # Get list of potential next steps
    def get_candidates(self, cycle, size, min_rate):
        if len(cycle) < size:  # generate candidates provided that they are within size
            res = list(map(lambda x: self.find_candidate(cycle, x, min_rate), self.reactions))
            return self.remove_val(res, None)
        else:
            temp_cycle = cycle.copy()
            temp_cycle.append(cycle[0])
            self.nuc_change[0] = cycle[0].N - cycle[-1].N
            self.nuc_change[1] = cycle[0].Z - cycle[-1].Z
            self.nuc_change[2] = cycle[0].A - cycle[-1].A
            i = self.check_priority()
            elem = self.rates[np.where(self.rates[:, 0] == ion(temp_cycle[-2 + i]))]
            if len(elem) > 0 and self.nuc_change in self.reactions and \
                    self.str_to_float(elem[0, self.reactions.index(self.nuc_change) + 1]) > min_rate:
                # you could change the threshold rate
                return [cycle[0]]
            else:
                return []

    # Recursively, perform a depth-first search to find valid solutions
    def search(self, cycle, size, min_rate, solutions):
        # Check is the state is valid
        if self.is_valid_state(cycle):
            # Add a copy of the valid state to list of solutions
            solutions.append(cycle.copy())
            # return # uncomment if you only need to find one valid solution
        # Iterate through the candidates that can be used
        # to construct the next state
        for candidate in self.get_candidates(cycle, size, min_rate):
            # Add candidate to the current state
            cycle.append(candidate)
            # Call search function with updated state
            self.search(cycle, size, min_rate, solutions)
            # Remove the current candidate from the current state
            cycle.pop(-1)

    # Entry point to the program
    # responsible for returning the valid solutions
    def solve(self, initial_elem, size, min_rate):
        # start with an empty list of solutions
        solutions = []
        # start with an empty state
        cycle = [ion(initial_elem)]
        # initiate the recursive search
        self.search(cycle, size, min_rate, solutions)
        # return the final list of solutions
        return solutions

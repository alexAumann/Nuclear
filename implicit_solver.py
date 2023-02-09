import re
import numpy as np
import matplotlib.pyplot as plt
from isotope import ion
from nuclear import Nuclear


class ImplicitSolver(Nuclear):

    def __init__(self, data):
        Nuclear.__init__(self, data)  # is this right?
        self.nuc_change = [0, 0, 0]
        self.particle_indices = [0, 0, 0]

    # cycle abundances as {a: x1, p: x2, n: x3, C12: x4, ...}
    # need to check if this works
    # most likely better implementation

    # this assumes simple reactions as of present
    def create_jacobian(self, cycle, abu_cycle):
        ordering = self.order_iso(cycle)
        # + 3 since we want to look at n, p and alpha
        # The order of the matrix will be [n, p, alpha, self.order_iso(cycle)[0], ...]
        rate_mat = np.zeros((len(ordering) + 3, len(ordering) + 3))
        rates = np.delete(np.array(self.get_rates()[1:]), [3, 4], 1)
        for n in range(len(ordering)):
            # looks at the change in nucleus (final - initial)
            self.nuc_change[0] = ion(cycle[n + 1]).N - ion(cycle[n]).N
            self.nuc_change[1] = ion(cycle[n + 1]).Z - ion(cycle[n]).Z
            self.nuc_change[2] = ion(cycle[n + 1]).A - ion(cycle[n]).A
            # checks for lower Z, then lower A
            # looks at the next isotope for reaction information with i = 1
            priority = self.check_priority()
            for elem in rates:  # I could extract the rates?
                if elem[0] == ion(cycle[n + priority]):

                    if self.reactions.index(self.nuc_change) == 0 or self.reactions.index(self.nuc_change) == 1:
                        self.ng_helper(rate_mat, ordering, self.nuc_change, elem, cycle, abu_cycle, n)

                    elif self.reactions.index(self.nuc_change) == 2 or self.reactions.index(self.nuc_change) == 3:
                        self.beta_helper(rate_mat, ordering, self.nuc_change, elem, cycle, abu_cycle, n)

                    elif self.reactions.index(self.nuc_change) == 4 or self.reactions.index(self.nuc_change) == 5:
                        self.pg_helper(rate_mat, ordering, self.nuc_change, elem, cycle, abu_cycle, n)

                    elif self.reactions.index(self.nuc_change) == 6 or self.reactions.index(self.nuc_change) == 7:
                        self.ap_helper(rate_mat, ordering, self.nuc_change, elem, cycle, abu_cycle, n)

                    elif self.reactions.index(self.nuc_change) == 8 or self.reactions.index(self.nuc_change) == 9:
                        self.an_helper(rate_mat, ordering, self.nuc_change, elem, cycle, abu_cycle, n)

                    elif self.reactions.index(self.nuc_change) == 10 or self.reactions.index(self.nuc_change) == 11:
                        self.ag_hepler(rate_mat, ordering, self.nuc_change, elem, cycle, abu_cycle, n)

                    break
        return rate_mat

    def create_vector(self, cycle, abu_cycle):
        ordering = self.order_iso(cycle)
        # + 3 since we want to look at n, p and alpha
        # The order of the matrix will be [n, p, alpha, self.order_iso(cycle)[0], ...]
        vec = np.zeros((len(ordering) + 3, 1))
        rates = np.delete(np.array(self.get_rates()[1:]), [3, 4], 1)
        for n in range(len(ordering)):
            # looks at the change in nucleus (final - initial)
            self.nuc_change[0] = ion(cycle[n + 1]).N - ion(cycle[n]).N
            self.nuc_change[1] = ion(cycle[n + 1]).Z - ion(cycle[n]).Z
            self.nuc_change[2] = ion(cycle[n + 1]).A - ion(cycle[n]).A
            # checks for lower Z, then lower A
            # looks at the next isotope for reaction information with i = 1
            priority = self.check_priority()
            for elem in rates:  # I could extract the rates?
                if elem[0] == ion(cycle[n + priority]):

                    if self.reactions.index(self.nuc_change) == 0 or self.reactions.index(self.nuc_change) == 1:
                        self.ng_vec(vec, ordering, self.nuc_change, elem, cycle, abu_cycle, n)

                    elif self.reactions.index(self.nuc_change) == 2 or self.reactions.index(self.nuc_change) == 3:
                        self.beta_vec(vec, ordering, self.nuc_change, elem, cycle, abu_cycle, n)

                    elif self.reactions.index(self.nuc_change) == 4 or self.reactions.index(self.nuc_change) == 5:
                        self.pg_vec(vec, ordering, self.nuc_change, elem, cycle, abu_cycle, n)

                    elif self.reactions.index(self.nuc_change) == 6 or self.reactions.index(self.nuc_change) == 7:
                        self.ap_vec(vec, ordering, self.nuc_change, elem, cycle, abu_cycle, n)

                    elif self.reactions.index(self.nuc_change) == 8 or self.reactions.index(self.nuc_change) == 9:
                        self.an_vec(vec, ordering, self.nuc_change, elem, cycle, abu_cycle, n)

                    elif self.reactions.index(self.nuc_change) == 10 or self.reactions.index(self.nuc_change) == 11:
                        self.ag_vec(vec, ordering, self.nuc_change, elem, cycle, abu_cycle, n)

                    break
        return vec

    def numerical_sol(self, cycle, abu_cycle, time_step, total_time):
        ordering = ["n", "p", "a"]
        ordering += self.order_iso(cycle)
        temp = []
        sol = []
        time = [0]
        time_mat = np.zeros((len(abu_cycle), len(abu_cycle)))
        for i in range(len(abu_cycle)):
            time_mat[i][i] = 1 / time_step
        for i in ordering:
            temp.append(abu_cycle[i])
        sol.append(temp.copy())
        for i in range(int(np.floor(total_time / time_step))):
            change = np.linalg.solve(time_mat-self.create_jacobian(cycle, abu_cycle), self.create_vector(cycle, abu_cycle))
            for j in range(len(ordering)):
                temp[j] += change[j]
                abu_cycle[ordering[j]] += change[j]
            sol += temp.copy()[0]
            time.append(time[-1] + 1/2)
        return sol, time

    # nuc_change[2] == 1 or - 1

    def ng_vec(self, vec, ordering, nuc_change, elem, cycle, abu_cycle, n):
        if nuc_change[0] == 1:
            # neutron rates
            vec[0, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["n"]
            # reactant rates
            vec[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["n"]

            vec[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["n"]

        elif nuc_change[0] == -1:
            # neutron rates
            vec[0, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            # reactant rates
            vec[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]

            vec[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]

    def beta_vec(self, vec, ordering, nuc_change, elem, cycle, abu_cycle, n):
        if nuc_change[0] == 1:

            vec[0, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["n"]

            vec[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["n"]

            vec[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["n"]

            vec[1, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["n"]

        elif nuc_change[0] == -1:

            vec[1, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["p"]

            vec[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["p"]

            vec[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["p"]

            vec[0, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["p"]

    def pg_vec(self, vec, ordering, nuc_change, elem, cycle, abu_cycle, n):
        if nuc_change[1] == 1:
            # neutron rates
            vec[1, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["p"]
            # reactant rates
            vec[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["p"]

            vec[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["p"]

        elif nuc_change[1] == -1:
            # neutron rates
            vec[1, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            # reactant rates
            vec[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]

            vec[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]

    def an_vec(self, vec, ordering, nuc_change, elem, cycle, abu_cycle, n):
        if nuc_change[2] == 3:
            vec[2, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["a"]

            vec[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["a"]

            vec[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["a"]

            vec[0, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["a"]

        elif nuc_chnage[2] == -3:

            vec[0, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["n"]

            vec[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["n"]

            vec[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["n"]

            vec[2, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["n"]

    def ap_vec(self, vec, ordering, nuc_change, elem, cycle, abu_cycle, n):
        if nuc_change[2] == 3:
            vec[2, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["a"]

            vec[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["a"]

            vec[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["a"]

            vec[1, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["a"]

        elif nuc_change[2] == -3:
            vec[1, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["p"]

            vec[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["p"]

            vec[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["p"]

            vec[2, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["p"]

    def ag_vec(self, vec, ordering, nuc_change, elem, cycle, abu_cycle, n):
        if nuc_change[2] == 4:
            vec[2, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["a"]

            vec[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["a"]

            vec[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]] * abu_cycle["a"]

        elif nuc_chnage[2] == -4:
            vec[2, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]

            vec[ordering.index(cycle[n]) + 3, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]

            vec[ordering.index(cycle[n + 1]) + 3, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]

    def ng_helper(self, rate_mat, ordering, nuc_change, elem, cycle, abu_cycle, n):
        if nuc_change[0] == 1:
            # neutron rates
            rate_mat[0, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[0, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["n"]
            # reactant rates
            rate_mat[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n]) + 3, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["n"]

            rate_mat[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n + 1]) + 3, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["n"]

        elif nuc_change[0] == -1:
            # neutron rates
            rate_mat[0, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1])
            # reactant rates
            rate_mat[ordering.index(cycle[n]) + 3, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1])

            rate_mat[ordering.index(cycle[n + 1]) + 3, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1])

    def beta_helper(self, rate_mat, ordering, nuc_change, elem, cycle, abu_cycle, n):
        if nuc_change[0] == 1:

            rate_mat[0, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[0, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["n"]

            rate_mat[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n]) + 3, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["n"]

            rate_mat[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n + 1]) + 3, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["n"]

            rate_mat[1, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[1, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["n"]

        elif nuc_change[0] == -1:

            rate_mat[1, 1] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[0, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["p"]

            rate_mat[ordering.index(cycle[n]) + 3, 1] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n]) + 3, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["p"]

            rate_mat[ordering.index(cycle[n + 1]) + 3, 1] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n + 1]) + 3, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["p"]

            rate_mat[0, 1] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[0, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["p"]

    def pg_helper(self, rate_mat, ordering, nuc_change, elem, cycle, abu_cycle, n):
        if nuc_change[1] == 1:
            # neutron rates
            rate_mat[1, 1] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[1, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["p"]
            # reactant rates
            rate_mat[ordering.index(cycle[n]) + 3, 1] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n]) + 3, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["p"]

            rate_mat[ordering.index(cycle[n + 1]) + 3, 1] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n + 1]) + 3, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["p"]

        elif nuc_change[1] == -1:
            # neutron rates
            rate_mat[1, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1])
            # reactant rates
            rate_mat[ordering.index(cycle[n]) + 3, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1])

            rate_mat[ordering.index(cycle[n + 1]) + 3, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1])

    def ap_helper(self, rate_mat, ordering, nuc_change, elem, cycle, abu_cycle, n):
        if nuc_change[2] == 3:
            rate_mat[2, 2] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[2, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["a"]

            rate_mat[ordering.index(cycle[n]) + 3, 2] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n]) + 3, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["a"]

            rate_mat[ordering.index(cycle[n + 1]) + 3, 2] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n + 1]) + 3, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["a"]

            rate_mat[1, 2] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[1, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["a"]

        elif nuc_change[2] == -3:

            rate_mat[1, 1] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[1, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["p"]

            rate_mat[ordering.index(cycle[n]) + 3, 1] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n]) + 3, ordering.index(cycle[n]) + 3] = -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["p"]

            rate_mat[ordering.index(cycle[n + 1]) + 3, 1] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n + 1]) + 3, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["p"]

            rate_mat[2, 1] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[2, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["p"]

    def an_helper(self, rate_mat, ordering, nuc_change, elem, cycle, abu_cycle, n):
        if nuc_change[2] == 3:
            rate_mat[2, 2] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[2, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["a"]

            rate_mat[ordering.index(cycle[n]) + 3, 2] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n]) + 3, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["a"]

            rate_mat[ordering.index(cycle[n + 1]) + 3, 2] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n + 1]) + 3, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["a"]

            rate_mat[0, 2] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[0, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["a"]

        elif nuc_chnage[2] == -3:

            rate_mat[0, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[0, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["n"]

            rate_mat[ordering.index(cycle[n]) + 3, 0] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n]) + 3, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["n"]

            rate_mat[ordering.index(cycle[n + 1]) + 3, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n + 1]) + 3, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["n"]

            rate_mat[2, 0] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[2, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["n"]

    def ag_helper(self, rate_mat, ordering, nuc_change, elem, cycle, abu_cycle, n):
        if nuc_change[2] == 4:
            rate_mat[2, 2] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[2, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["a"]

            rate_mat[ordering.index(cycle[n]) + 3, 2] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n]) + 3, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["a"]

            rate_mat[ordering.index(cycle[n + 1]) + 3, 2] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle[cycle[n]]
            rate_mat[ordering.index(cycle[n + 1]) + 3, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1]) * abu_cycle["a"]

        elif nuc_chnage[2] == -4:
            rate_mat[2, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1])

            rate_mat[ordering.index(cycle[n]) + 3, ordering.index(cycle[n]) + 3] += -self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1])

            rate_mat[ordering.index(cycle[n + 1]) + 3, ordering.index(cycle[n]) + 3] += self.str_to_float(
                elem[self.reactions.index(self.nuc_change) + 1])

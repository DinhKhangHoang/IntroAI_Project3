import numpy as np
from functools import reduce


class FactorNode:
    def __init__(self, factors, probabilities, node=None, parents=None, domain=None):
        self.node = [node]
        self.parents = parents
        self.factors = factors
        self.domain = domain
        self.prob = probabilities


class BayesianNetwork:
    def __init__(self, filename):

        f = open(filename, 'r')
        N = int(f.readline())
        self.factors = {}

        lines = f.readlines()
        for line in lines:
            node, parents, domain, shape, probabilities = self.__extract_model(
                line)
            self.factors[node] = FactorNode(
                parents+[node], probabilities, node, parents, domain)
        f.close()

    def exact_inference(self, filename):
        results_prob = 0
        f = open(filename, 'r')
        queryVar, evidenceVar = self.__extract_query(
            f.readline())
        f.close()

        lstVars = list(queryVar.keys()) + \
            list(evidenceVar.keys())

        factors = {}
        lst = reduce(lambda x, y: x + [y] +
                     self.factors[y].parents, lstVars, [])
        dependVars = list(set(lst))
        for var in dependVars:
            factors[var] = self.factors[var]

        hidden_var = list(
            factors.keys()-queryVar.keys()-evidenceVar.keys())

        for var in hidden_var:
            self.remove_hiddenVars(var[0], factors)

        probTable, finalFactor = self.join_factors(
            list(factors.items()))

        finalFactor_axes = list(
            map(lambda var: finalFactor.index(var), list(evidenceVar.keys()) + list(queryVar.keys())))

        results_prob = np.transpose(probTable, finalFactor_axes)

        results_prob = reduce(
            lambda result, evidence: result[self.factors[evidence[0]].domain.index(evidence[1])], evidenceVar.items(), results_prob)

        results_prob = results_prob/np.sum(results_prob)

        results_prob = reduce(
            lambda result, query: result[self.factors[query[0]].domain.index(query[1])], queryVar.items(), results_prob)
        return results_prob

    def remove(self, lst, i):
        lst.remove(i)
        return lst

    def join_factors(self, listNodes):
        result_node = listNodes[0]
        result_node_factors = result_node[1].factors
        result_node_prob = result_node[1].prob
        for node in listNodes[1:]:
            next_node_factors = node[1].factors
            next_node_prob = node[1].prob
            node_intersect = list(
                set(result_node_factors).intersection(set(next_node_factors)))

            result_node_variable_intersect_index = list(
                map(lambda x: result_node_factors.index(x), node_intersect))

            result_node_axes = list(range(result_node_prob.ndim))

            result_node_axes = reduce(lambda lst, i: self.remove(
                lst, i), result_node_variable_intersect_index, result_node_axes)

            result_node_axes += result_node_variable_intersect_index

            next_node_variable_intersect_index = list(
                map(lambda x: next_node_factors.index(x), node_intersect))
            next_node_axes = list(range(next_node_prob.ndim))

            next_node_axes = reduce(lambda lst, i: self.remove(
                lst, i), next_node_variable_intersect_index, next_node_axes)
            next_node_axes = next_node_variable_intersect_index + next_node_axes

            result_node_prob = np.transpose(
                result_node_prob, result_node_axes)
            next_node_prob = np.transpose(
                next_node_prob, next_node_axes)

            new_factor_dimension = len(
                set(result_node_factors).union(set(next_node_factors)))

            for _ in range(new_factor_dimension-len(result_node_axes)):
                result_node_prob = np.expand_dims(result_node_prob, -1)

            result_node_prob = result_node_prob*next_node_prob
            for x in node_intersect:
                result_node_factors.remove(x)
                next_node_factors.remove(x)

            result_node_factors = result_node_factors + node_intersect + next_node_factors

        return result_node_prob, result_node_factors

    def remove_hiddenVars(self, var_name, nodes):
        listNodes = []
        for key, factor in nodes.items():
            if var_name in factor.factors:
                listNodes.append([key, factor])
        for x in listNodes:
            nodes.pop(x[0])

        result_node_prob, result_node_factors = self.join_factors(
            listNodes)
        var_index = result_node_factors.index(var_name)
        result_node_factors.remove(var_name)
        prob = np.sum(result_node_prob, axis=var_index)

        nodes[tuple(result_node_factors)] = FactorNode(
            result_node_factors, prob)

    def approx_inference(self, filename):
        result = 0
        f = open(filename, 'r')
        # YOUR CODE HERE

        f.close()
        return result

    def __extract_model(self, line):
        parts = line.split(';')
        node = parts[0]
        if parts[1] == '':
            parents = []
        else:
            parents = parts[1].split(',')
        domain = parts[2].split(',')
        shape = eval(parts[3])
        probabilities = np.array(eval(parts[4])).reshape(shape)
        return node, parents, domain, shape, probabilities

    def __extract_query(self, line):
        parts = line.split(';')

        # extract query variables
        query_variables = {}
        for item in parts[0].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            query_variables[lst[0]] = lst[1]

        # extract evidence variables
        evidence_variables = {}
        for item in parts[1].split(','):
            if item is None or item == '':
                continue
            lst = item.split('=')
            evidence_variables[lst[0]] = lst[1]
        return query_variables, evidence_variables

# type: ignore

import hypothesis
import hypothesis.strategies as hs
import os
import random
import string
import unittest

from heuristics import ge_run
from heuristics import ge_helpers
from heuristics import population
from heuristics import grammar
from fitness import fitness


def get_run_param():
    param = hs.fixed_dictionaries(
        {
            "population_size": hs.integers(min_value=2, max_value=6),
            "max_length": hs.integers(min_value=30, max_value=40),
            "codon_size": hs.integers(min_value=10, max_value=20),
            "elite_size": hs.integers(min_value=0, max_value=2),
            "generations": hs.integers(min_value=1, max_value=4),
            "tournament_size": hs.integers(min_value=1, max_value=3),
            "seed": hs.integers(min_value=0, max_value=10),
            "crossover_probability": hs.floats(min_value=0.0, max_value=1.0),
            "mutation_probability": hs.floats(min_value=0.0, max_value=1.0),
            "integer_input_element_max": hs.integers(min_value=1, max_value=1000),
            "bnf_grammar": hs.just("tests/grammars/symbolic_regression.bnf"),
            "fitness_function": hs.just(
                {
                    "name": "SRExpression",
                    "symbolic_expression": "None",
                    "exemplars": "[(x1, x2, x1**2 + x2**2) for x1, x2 in \
                                  zip(range(-11, 1), range(0, 10))]",
                }
            ),
        }
    )
    return param


def get_individual():
    population.Individual.max_length = hs.integers(min_value=1, max_value=128)
    population.Individual.codon_size = hs.integers(min_value=1, max_value=128)
    individual = hs.builds(population.Individual, hs.just(None))
    return individual


def get_individual_with_fitness(genome, codon_size, max_length, fit, used_input=None):
    population.Individual.codon_size = codon_size
    population.Individual.max_length = max_length
    individual = population.Individual(genome)
    individual.fitness = fit
    individual.used_input = used_input
    return individual


def get_individuals(n=None, used_input=hs.just(None)):
    if n is None:
        n = hs.integers(min_value=1, max_value=20)

    individuals = n.flatmap(
        lambda x: hs.lists(
            hs.builds(
                get_individual_with_fitness,
                hs.just(None),
                hs.integers(min_value=1, max_value=128),
                hs.integers(min_value=1, max_value=128),
                hs.integers(),
                used_input,
            ),
            max_size=x,
            min_size=x,
        )
    )

    return individuals


def get_generational_replacement():
    n = hs.integers(min_value=1, max_value=100)
    _n = hs.shared(n, "same")
    _m = hs.shared(n, "same")
    _ps = hs.shared(n, "same")
    _es = hs.shared(n, "same")
    elite_size = _es.flatmap(lambda x: hs.integers(min_value=0, max_value=x - 1))
    new_population = get_individuals(_n)
    old_population = get_individuals(_m)

    return hs.fixed_dictionaries(
        {
            "new_population": new_population,
            "old_population": old_population,
            "population_size": _ps,
            "elite_size": elite_size,
        }
    )


def get_tournament_selection():
    n = hs.integers(min_value=1, max_value=100)
    _n = hs.shared(n, "same")
    _m = hs.shared(n, "same")
    population_size = hs.shared(n, "same")
    tournament_size = _m.flatmap(lambda x: hs.integers(min_value=1, max_value=x))
    pop = _n.flatmap(lambda x: get_individuals(hs.just(x)))

    return hs.fixed_dictionaries(
        {"population": pop, "population_size": population_size, "tournament_size": tournament_size}
    )


def get_int_flip_mutation():
    mutation_probability = hs.floats(min_value=0.0, max_value=1.0)
    codon_size = hs.integers(min_value=1)
    individual = hs.builds(
        get_individual_with_fitness,
        hs.just(None),
        hs.integers(min_value=1, max_value=128),
        hs.integers(min_value=1, max_value=128),
        hs.integers(),
    )

    return hs.fixed_dictionaries(
        {
            "individual": individual,
            "mutation_probability": mutation_probability,
            "codon_size": codon_size,
        }
    )


def get_variation():
    param = hs.fixed_dictionaries(
        {
            "population_size": hs.integers(min_value=1, max_value=100),
            "codon_size": hs.integers(min_value=1, max_value=100),
            "crossover_probability": hs.floats(min_value=0.0, max_value=1.0),
            "mutation_probability": hs.floats(min_value=0.0, max_value=1.0),
            "population": get_individuals(
                hs.integers(min_value=2, max_value=100),
                used_input=hs.integers(min_value=1, max_value=100),
            ),
        }
    )

    return param


def write_rule_str(n_non_terminals, n_terminals):
    def get_characters(alphabet, _n=None):
        if _n is None:
            _n = random.randint(0, 10)
        _ws = [random.choice(alphabet) for _ in range(_n)]
        _str = "".join(_ws)

        return _str

    def get_non_terminal(size):
        alphabet = string.digits + string.ascii_letters + "-_.,:"
        _str = "<%s>" % (get_characters(alphabet, size))
        return _str

    def get_terminal(size):
        alphabet = string.digits + string.ascii_letters + "_.,: "
        _str = get_characters(alphabet, size)
        if len(_str.strip()) == 0:
            _str = _str + "a"
        return _str

    def get_rule(n_productions, lhs, terminals, non_terminals):
        def get_whitespaces():
            _ws = get_characters(["\t", " "])
            _str = "".join(_ws)
            return _str

        rule_separator = grammar.Grammar.rule_separator
        strs = ""
        for i in range(n_productions):
            production = []
            n_symbols = random.randint(1, 10)
            for _ in range(n_symbols):
                try:
                    if random.random() < 0.5:
                        symbol = random.choice(terminals)
                    else:
                        symbol = random.choice(non_terminals)
                except IndexError:
                    symbol = ""

                production.append(symbol)

            # TODO efficient string building in python...
            production = "".join(production)
            if i < (n_productions - 1):
                production_separator = grammar.Grammar.production_separator
            else:
                production_separator = ""

            strs = "%s%s%s%s%s" % (
                strs,
                production,
                get_whitespaces(),
                production_separator,
                get_whitespaces(),
            )
        _str = "%s%s%s%s%s" % (
            lhs,
            get_characters(["\t", " "]),
            rule_separator,
            get_characters(["\t", " "]),
            strs,
        )

        return _str

    non_terminals = []
    for _ in range(n_non_terminals):
        size = random.randint(1, 10)
        nt = get_non_terminal(size)
        if nt not in non_terminals:
            non_terminals.append(nt)

    terminals = []
    for _ in range(n_terminals):
        size = random.randint(1, 10)
        t = get_terminal(size)
        if t not in terminals:
            terminals.append(t)

    rules = []
    for nt in non_terminals:
        n_productions = random.randint(1, 4)
        rule = get_rule(n_productions, nt, terminals, non_terminals)
        rules.append(rule)

    _str = "\n".join(rules)

    return {
        "bnf_string": _str,
        "terminals": terminals,
        "non_terminals": non_terminals,
        "rules": rules,
    }


def get_bnf_string():
    bnf_string = hs.builds(
        write_rule_str,
        hs.integers(min_value=1, max_value=10),
        hs.integers(min_value=1, max_value=10),
        hs.random_module(),
    )

    return bnf_string


def get_grammar():
    def _get_grammar(n_non_terminals, n_terminals):
        gramm = grammar.Grammar("")
        bnf_string = write_rule_str(n_non_terminals, n_terminals)
        gramm.parse_bnf_string(bnf_string["bnf_string"])

        return gramm

    gramm2 = hs.builds(
        _get_grammar,
        hs.integers(min_value=1, max_value=10),
        hs.integers(min_value=1, max_value=10),
    )
    inputs = hs.lists(hs.integers(min_value=1))

    return hs.fixed_dictionaries({"grammar": gramm2, "inputs": inputs})


class TestMuleGE(unittest.TestCase):
    @hypothesis.given(individuals=get_individuals())
    def test_sort_population(self, individuals):
        sorted_individuals = ge_helpers.sort_population(individuals)
        for i in range(1, len(sorted_individuals)):
            self.assertGreaterEqual(
                sorted_individuals[i - 1].fitness, sorted_individuals[i].fitness
            )

    @hypothesis.given(get_generational_replacement())
    def test_generational_replacement(self, args):
        new_population = args["new_population"]
        old_population = args["old_population"]
        population_size = args["population_size"]
        elite_size = args["elite_size"]
        pop = ge_helpers.generational_replacement(
            new_population, old_population, elite_size, population_size
        )
        self.assertEqual(len(pop), population_size)
        # TODO Checking deep copy well enough?
        for i in range(population_size):
            self.assertIsNot(pop[i], old_population[i])

    @hypothesis.given(get_tournament_selection())
    def test_tournament_selection(self, args):
        pop = args["population"]
        population_size = args["population_size"]
        tournament_size = args["tournament_size"]
        selected_population = ge_helpers.tournament_selection(pop, population_size, tournament_size)
        self.assertEqual(len(selected_population), population_size)
        # TODO Checking reference well enough?
        for i in range(population_size):
            self.assertIn(selected_population[i], pop)

    @hypothesis.given(get_int_flip_mutation())
    def test_int_flip_mutation(self, args):
        individual = args["individual"]
        mutation_probability = args["mutation_probability"]
        population.Individual.codon_size = args["codon_size"]
        org_genome = individual.genome[:]
        new_individual = ge_helpers.int_flip_mutation(individual, mutation_probability)

        self.assertIs(new_individual, individual)
        self.assertIs(new_individual.genome, individual.genome)
        same = True
        cnt = 0
        while same and cnt < len(individual.genome):
            same = org_genome[cnt] == new_individual.genome[cnt]
            cnt += 1

        if not same:
            self.assertEqual(new_individual.phenotype, population.Individual.DEFAULT_PHENOTYPE)
            self.assertEqual(new_individual.used_input, 0)
            self.assertEqual(new_individual.fitness, fitness.DEFAULT_FITNESS)

    @hypothesis.given(get_variation())
    def test_variation(self, args):
        # TODO not pretty argument passing in get_variation...
        parents = args["population"]
        param = args
        new_population = ge_helpers.variation(parents, param, param["population_size"])
        self.assertEqual(len(new_population), param["population_size"])
        self.assertIsNot(new_population, parents)
        genomes = [_.genome for _ in parents]
        for i in range(len(new_population)):
            self.assertNotIn(new_population[i], parents)
            for genome in genomes:
                self.assertIsNot(
                    new_population[i].genome,
                    genome,
                    "%s in %s" % (new_population[i].genome, genome),
                )

    @hypothesis.given(bnf_string=get_bnf_string())
    def test_parse_bnf_string(self, bnf_string):
        hypothesis.assume(bnf_string != "")
        _bnf_string = bnf_string["bnf_string"]
        terminals = bnf_string["terminals"]
        terminals = [_.strip() for _ in terminals]
        non_terminals = bnf_string["non_terminals"]
        rules = bnf_string["rules"]
        gramm = grammar.Grammar("")
        gramm.parse_bnf_string(_bnf_string)
        self.assertEqual(gramm.start_rule[0], non_terminals[0])
        self.assertEqual(
            len(gramm.non_terminals),
            len(non_terminals),
            "%s != %s" % (gramm.non_terminals, non_terminals),
        )
        # TODO generate terminals so all generated terminals are
        # present in bnf_string
        for terminal in gramm.terminals:
            try:
                self.assertIn(terminal, terminals, "%s not in %s" % (terminal, terminals))
            except AssertionError:
                # When parsing terminals two or more subsequent
                # terminals can be parsed as one. Therefore we need to
                # check if the terminals exist in this terminal
                # TODO generate better terminals...
                matches = [_ for _ in terminals if _ in terminal]
                self.assertGreaterEqual(len(matches), 1)

        self.assertEqual(len(gramm.rules), len(rules))

    @hypothesis.given(params=get_grammar())
    def test_generate_sentence(self, params):
        gramm = params["grammar"]
        inputs = params["inputs"]
        output, used_input = gramm.generate_sentence(inputs)
        self.assertLessEqual(used_input, len(inputs))
        if output is not None:
            self.assertEqual(type(output) == str, True)
        else:
            self.assertIsNone(output)

    @hypothesis.given(params=get_grammar())
    def test_map_input_with_grammar(self, params):
        # TODO pass in individual instead of making it in function
        hypothesis.assume(len(params["inputs"]) > 0)
        gramm = params["grammar"]
        inputs = params["inputs"]
        individual = None
        try:
            population.Individual.max_length = len(inputs)
            population.Individual.codon_size = max(inputs) + 1
            individual = population.Individual(inputs)
            _individual = ge_helpers.map_input_with_grammar(individual, gramm)
            self.assertIs(individual, _individual)
            self.assertNotEqual(_individual.phenotype, population.Individual.DEFAULT_PHENOTYPE)
            self.assertGreater(_individual.used_input, -1)
        except ValueError:
            self.assertEqual(individual.phenotype, population.Individual.DEFAULT_PHENOTYPE)

    @hypothesis.given(param=get_run_param(), rnd=hs.random_module())
    def test_run(self, param):
        hypothesis.assume(param["tournament_size"] <= param["population_size"])
        hypothesis.assume(param["elite_size"] < param["population_size"])
        try:
            _individual_dict = ge_run.run(param, coev=False)
            _individual0 = _individual_dict[_individual_dict.keys()[0]]
            self.assertIsNotNone(_individual0.phenotype)
        except ValueError as e:
            self.assertEqual("Phenotype is None", str(e))

    @hypothesis.given(param=get_run_param(), rnd=hs.random_module())
    def test_search_loop(self, param):
        hypothesis.assume(param["tournament_size"] <= param["population_size"])
        hypothesis.assume(param["elite_size"] < param["population_size"])

        # TODO too much setup, can it be refactored...
        fitness_function = ge_run.get_fitness_function(param["fitness_function"])
        gramm = grammar.Grammar(param["bnf_grammar"])
        gramm.read_bnf_file(gramm.file_name)
        population.Individual.codon_size = param["codon_size"]
        population.Individual.max_length = param["max_length"]
        individuals = ge_helpers.initialise_population(param["population_size"])
        pop = population.Population(fitness_function, gramm, individuals)

        try:
            _individual = ge_helpers.search_loop(pop, param)
            self.assertIsNotNone(_individual.phenotype)
        except ValueError as e:
            self.assertEqual("Phenotype is None", str(e))

    def test_read_bnf_file(self):
        """Non hypothesis test, since we want to check known files...
        TODO write
        """
        _dir = "tests/grammars/"
        files = os.listdir(_dir)
        for _file in files:
            if _file.endswith(".bnf"):
                path = os.path.join(_dir, _file)
                gramm = grammar.Grammar(path)
                gramm.read_bnf_file(gramm.file_name)
                # TDOD make assert


if __name__ == "__main__":
    unittest.main()

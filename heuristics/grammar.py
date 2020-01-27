#! /usr/bin/env python
import collections
import re
from typing import List, Tuple, Set

from heuristics.population import Individual


class Grammar(object):
    """
    Context Free Grammar. Symbols are tuples with (value, type),
    type is Terminal or NonTerminal
    """

    NT: str = "NT"  # Non Terminal
    T: str = "T"  # Terminal
    rule_separator: str = "::="
    production_separator: str = "|"

    def __init__(self, file_name: str) -> None:
        """Context free grammar.

        :param file_name: grammar file
        :type file_name: str
        """

        # due to error in pylint, throws unsubscriptable-object for the typing annotation
        self.rules: collections.OrderedDict[  # pylint: disable=unsubscriptable-object
            str, List[List[Tuple[str, str]]]
        ] = collections.OrderedDict()
        # TODO use an ordered set
        self.non_terminals: Set[str] = set()
        self.terminals: Set[str] = set()
        self.start_rule: Tuple[str, str] = ("", "")
        self.file_name: str = file_name

    def read_bnf_file(self, file_name: str) -> None:
        """Read a grammar file in BNF format. Wrapper for file reading.

        :param file_name: BNF grammar file
        :type file_name: str
        """
        assert file_name.endswith(".bnf")

        # Read the grammar file
        with open(file_name, "r") as in_file:
            lines: str = in_file.read()

        self.parse_bnf_string(lines)

    def parse_bnf_string(self, all_lines: str) -> None:
        """Parse a BNF string with REGEXP.

        # TODO use a non-regexp parser
        # TODO does not handle newlines well

        :param all_lines: BNF grammar
        :type all_lines: str

        """
        assert all_lines != ""
        _lines = all_lines
        non_terminal_pattern = re.compile(
            r"""(# Group  so `split()` returns all NTs and Ts.
                 # Do not allow space in NTs. Use lookbehind to match "<"
                 # and ">" only if not preceded by backslash.
                 (?<!\\)<\S+?(?<!\\)>
                 )""",
            re.VERBOSE,
        )
        production_separator_regex = re.compile(
            r"""# Use lookbehind to match "|" if not preceded by
                # backslash. `split()` returns only the productions.
                (?<!\\)\|""",
            re.VERBOSE,
        )
        # Left Hand Side(lhs) of rule
        lhs = None

        # Remember last character on line to handle multi line rules
        last_character = None
        lines: List[str] = all_lines.split("\n")
        for line in lines:
            line = line.strip()
            if not line.startswith("#") and line != "":
                # Split rules.
                rule_separators = line.count(Grammar.rule_separator)
                assert rule_separators < 2
                if rule_separators == 1:
                    lhs, productions = line.split(Grammar.rule_separator, 1)
                    lhs = lhs.strip()
                    assert len(lhs) > 2
                    assert non_terminal_pattern.search(lhs), "lhs is not a NT: {}".format(lhs)
                    self.non_terminals.add(lhs)
                    if self.start_rule[0] == "" and self.start_rule[1] == "":
                        self.start_rule = (lhs, self.NT)

                else:
                    productions = line

                assert productions != "", "{}\n{}\n{}".format(line, lines, _lines)

                # Find terminals and non-terminals
                tmp_productions = []
                production_split = production_separator_regex.split(productions)
                for production in production_split:
                    production = production.strip().replace(r"\|", Grammar.production_separator)
                    tmp_production = []
                    for symbol in non_terminal_pattern.split(production):
                        symbol = symbol.replace(r"\<", "<").replace(r"\>", ">")
                        if not symbol:
                            continue
                        elif non_terminal_pattern.match(symbol):
                            tmp_production.append((symbol, self.NT))
                        else:
                            self.terminals.add(symbol)
                            tmp_production.append((symbol, self.T))

                    if tmp_production:
                        tmp_productions.append(tmp_production)

                assert lhs is not None, "No lhs: {}\n{}".format(line, lines)

                # Create a rule
                if lhs not in self.rules:
                    self.rules[lhs] = tmp_productions
                else:
                    if len(production_split) > 1 or last_character == Grammar.production_separator:
                        self.rules[lhs].extend(tmp_productions)
                # TODO does not handle multiline terminals...

                # Remember the last character of the line
                last_character = productions[-1]

    def __str__(self) -> str:
        return "T:{}\nNT:{}\nR:{}\nS:{}\n".format(
            self.terminals, self.non_terminals, self.rules, self.start_rule
        )

    def generate_sentence(self, inputs: List[int]) -> Tuple[str, int]:
        """Map inputs via rules to output sentence (phenotype).

        :param inputs: Inputs used to generate sentence with grammar
        :type inputs: list of int
        :returns: Sentence and number of input used (phenotype)
        :rtype: tuple of str and int
        """
        used_input = 0
        # TODO faster data structure? E.g. queue
        output: List[str] = []
        # Needed to avoid infinite loops with poorly specified
        # grammars
        cnt = 0
        break_out = len(inputs) * len(self.terminals)
        unexpanded_symbols: List[Tuple[str, str]] = [self.start_rule]
        while unexpanded_symbols and used_input < len(inputs) and cnt < break_out:
            # Expand a production
            current_symbol: Tuple[str, str] = unexpanded_symbols.pop(0)
            # Set output if it is a terminal
            if current_symbol is not None and current_symbol[1] != Grammar.NT:
                output.append(current_symbol[0])
            else:
                production_choices = self.rules[current_symbol[0]]
                # Select a production
                current_production = inputs[used_input] % len(production_choices)
                # Use an input if there was more then 1 choice
                if len(production_choices) > 1:
                    used_input += 1

                # Derivation order is left to right(depth-first)
                unexpanded_symbols = production_choices[current_production] + unexpanded_symbols

            cnt += 1

        # Not fully expanded
        if unexpanded_symbols:
            return Individual.DEFAULT_PHENOTYPE, used_input
        else:
            str_output: str = "".join(output)
            return str_output, used_input

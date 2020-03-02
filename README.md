# donkey_ge

Simple implementation of Grammatical Evolution. Uses `python3`. 

## Install

Install requirements
```
pip install -r requirements.txt
```

## Run

Paths are relative the repository root.

### Tutorials

See [tutorials](tutorials). The [tests](tests) can also help understand the program.

### Simple evolutionary search

A population can be evolved against a single individual. For example, in an iterated prisoner's dilemma, a population of players can play against a player who is Always Defect.

```
python main.py -f tests/configurations/simple_configs/iterated_prisoners_dilemma.yml -o results
```

### Single population co-evolutionary search

A population can be evolved against itself, with players from the population playing others from the same population.

```
python main.py -f tests/configurations/singlepop_coev_configs/singlepop_ipd.yml -o results --coev
```

### Two way evolutionary search - Coevolutionary

Two or more populations with different strategies may also be evolved against each other.

```
python main.py -f tests/configurations/multipop_coev_configs/coevolution_nonintrusive_hawk_dove.yml -o results --coev
```

### Spatial evolutionary search

Finally, a population may play against itself in a spatial setting. A variety of pre-built graphs are available in the library. Players play their neighbors on the graph in this deterministic game.

```
python main.py -f tests/spatial_configs/multipop_coev_configs/spatial_intrusive_hd.yml.yml -o results --spatial
```

### `donkey_ge` output

`donkey_ge` prints some information to `stdout` regarding `settings` and
search progress for each iteration, see `donkey_ge.py:print_stats`. 

The output files have each generation as a list element, and each individual separated by a `,`, except where otherwise specified. Spatial games have special output. These data are written to a variety of output files depending on the settings.

### Usage
```
usage: main.py [-h] -f CONFIGURATION_FILE [-o OUTPUT_DIR] [--coev] [--spatial]

Run donkey_ge

optional arguments:
  -h, --help            show this help message and exit
  -f CONFIGURATION_FILE, --configuration_file CONFIGURATION_FILE
                        YAML configuration file. E.g.
                        configurations/demo_ge.yml
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Path to directory for output files. E.g.
                        donkey_ge_output
  --coev                Coevolution
  --spatial             Spatial
```

### Settings

Configurations are in `.yml` format, see examples in folder [configurations](tests/configurations).

Grammar is in *Backus-Naur Form (BNF)*, see examples in folder [grammars](tests/grammars)

## Test

Tests are in `tests` folder. E.g. run with `pytest`
```
pytest tests
```

Some tests are written with *Hypothesis*, http://hypothesis.readthedocs.io/en/master/index.html

## Development

Use `pre-commit.sh` as a pre-commit hook. E.g. `ln -s ../../pre-commit.sh .git/hooks/pre-commit`

## Documentation

See [docs/README.md](docs/README.md) for more details and basic
Evolutionary Computation background.

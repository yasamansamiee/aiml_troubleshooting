# Welcome to kuberspatiotemporal

A package to experiment with probabilistic, heterogeneous mixture models for risk analysis.

## Installation and development


First make sure to install Python (>= 3.7) the dependency management
tool [Poetry](https://python-poetry.org/) then create an isolated virtual
environment and install the dependencies.

```sh
poetry install 
```

Per terminal session,  the following command should be executed
to activate the virtual environment.

```sh
poetry shell
```

To work with [jupyter notebooks](https://jupyter.org), first register the
virtual environment (only once) and start a browser based notebook server.

```sh
python -m ipykernel install --user --name python_kuberspatiotemporal --display-name "Python3 (kuberspatiotemporal)"
jupyter notebook
```

To generate the documentation run:

```sh
cd doc/
make api # optional, only when the code base changes
make html
```

To run unit tests, run:

```sh
pytest --log-level=WARNING
# Specify a selected test
pytest --log-level=DEBUG -k "TestKuberModel"
```



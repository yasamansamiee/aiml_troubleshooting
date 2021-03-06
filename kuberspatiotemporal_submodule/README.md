![](https://github.com/acceptto-corp/kuberspatiotemporal/blob/add-badges/coverage.svg)
![](https://travis-ci.com/acceptto-corp/kuberspatiotemporal.svg?token=xvg6aZWm2phaynNQmftz&branch=master)

# Welcome to kuberspatiotemporal

A package to experiment with probabilistic, heterogeneous mixture models for risk analysis.

## Installation and development


First make sure to install Python (== 3.7) the dependency management
tool [Poetry](https://python-poetry.org/) then create an isolated virtual
environment and install the dependencies. Note that 3.8 currently
installation fails (issue with the dependencies) while the code itself is compatible.

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
pytest --log-level=DEBUG tests/test_large.py::TestLargeData::test_batch_unparam_em
```

To work with [VisualStudio Code](https://code.visualstudio.com/):

```sh
cp .vscode/template.settings.json .vscode/settings.json
which python # copy the path without the executable
```

and add the path to the virtual environment to in the `"python.pythonPath"` setting.

```sh
cp .vscode/template.settings.json .vscode/settings.json
which python # copy the path without the executable
```

and add the path to the virtual environment to in the `"python.pythonPath"` setting.

To use the classes, shortcuts have been added:

```python
from kuberspatiotemporal import (
    CompoundModel,
    KuberModel,
    SpatialModel
)
```

Build and read the documentation for more details about usage.


## TODOs and open issues

See generated documentation
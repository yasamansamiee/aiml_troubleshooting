FROM 064061306967.dkr.ecr.us-west-2.amazonaws.com/python-poetry-base

COPY . .

RUN replace_dep_tool.sh kuberspatiotemporal ./whl/kuberspatiotemporal*.whl pyproject.toml 

RUN poetry update --no-dev && \
    poetry install --no-dev && \
    poetry build && \
    cp -r dist/*.whl /opt/dist/
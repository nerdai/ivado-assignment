FROM python:3.11-slim as python
WORKDIR /

# Configure Poetry
ENV POETRY_VERSION=1.4.0
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

# Install dependencies
COPY poetry.lock pyproject.toml ./
RUN poetry install --no-interaction --no-cache --without dev --no-root 

# Copy source code and training data
COPY ./setup.sh /setup.sh
COPY ./ivado_assignment/ /ivado_assignment
COPY ./data/raw/2021-10-19_14-11-08_val_candidate_data.csv /data/raw/2021-10-19_14-11-08_val_candidate_data.csv
RUN sh setup.sh

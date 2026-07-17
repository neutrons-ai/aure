FROM python:3.11-slim

COPY pyproject.toml ./
COPY src/ ./src/

# Install the package, then remove git (only needed to fetch nr-isaac-format from GitHub)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && pip install --no-cache-dir ".[export]" \
    && apt-get purge -y git && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Working directory inside the container – users mount their data here
WORKDIR /work

ENTRYPOINT ["aure"]
CMD ["--help"]

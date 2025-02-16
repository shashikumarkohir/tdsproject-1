FROM python:3.12-slim-bookworm

# Install dependencies including Node.js and npm
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates nodejs npm && npm install -g prettier

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

COPY app.py /app

CMD ["uv", "run", "app.py"]
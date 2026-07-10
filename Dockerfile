FROM python:3.10-slim

# Add non-root user
RUN useradd -m -d /app -s /bin/bash aethelix

# Add Rust toolchain securely via multi-stage copy
COPY --from=rust:1.75-slim /usr/local/cargo /usr/local/cargo
COPY --from=rust:1.75-slim /usr/local/rustup /usr/local/rustup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

# Install system dependencies required for Rust compilation and Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire project
COPY . .

# Install Python dependencies and Maturin
RUN pip install --no-cache-dir -r requirements.txt maturin

# Build and install the Aethelix python package (including the Rust core via Maturin)
RUN pip install --no-cache-dir -e .

# Transfer ownership to non-root user for runtime safety
RUN chown -R aethelix:aethelix /app

USER aethelix

# Expose Streamlit default port
EXPOSE 8501

# Run the Mission Control Dashboard
ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

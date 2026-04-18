FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for Rust compilation and Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain to compile Aethelix core via Maturin
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python dependencies and Maturin
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt maturin

# Copy the entire project
COPY . .

# Build and install the Aethelix python package (including the Rust core via Maturin)
RUN pip install --no-cache-dir -e .

# Expose Streamlit default port
EXPOSE 8501

# Run the Mission Control Dashboard
ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

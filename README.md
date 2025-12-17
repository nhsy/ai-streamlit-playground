# Ollama Streamlit Playground

This is a Streamlit application that interfaces with a local Ollama instance.

## Prerequisites

- [Docker](https://www.docker.com/) installed on your machine.
- [Ollama](https://ollama.com/) running on your host machine.

## Installation

To install Ollama using Homebrew:

```bash
brew install ollama
```

## Managing Ollama

If you installed Ollama via Homebrew, you can manage the service with:

-   **Start Ollama**: `brew services start ollama`
-   **Stop Ollama**: `brew services stop ollama`

## Running with Docker

1.  Make sure Ollama is running on your host machine (usually port 11434).
2.  Build and run the container:

    ```bash
    docker-compose up --build
    ```

3.  Open your browser and navigate to `http://localhost:8501`.

## Automation (Taskfile)

This project uses [Task](https://taskfile.dev/) to automate common commands.

-   **Install dependencies**: `task install`
-   **Run locally**: `task run`
-   **Run tests**: `task test`
-   **Build Docker**: `task docker:build`
-   **Run Docker**: `task docker:up`
-   **Stop Docker**: `task docker:down`

## Development

The `docker-compose.yml` mounts the current directory to `/app` in the container, so changes to `app.py` will be reflected immediately (thanks to Streamlit's auto-reload).

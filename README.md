# AI Streamlit Playground

![Build Status](https://github.com/nhsy/ai-streamlit-playground/actions/workflows/test.yml/badge.svg)

This is a Streamlit application that interfaces with local Ollama models and IBM watsonx.ai cloud models.

## Features

- 🔄 **Dual Provider Support**: Switch between local Ollama and cloud-based IBM watsonx
- 💬 **Chat Interface**: Interactive chat with conversation history
- 🔄 **Text Transformation**: Apply templates for common text operations
- 📎 **File Upload**: Support for PDF, TXT, CSV, and more
- 🎛️ **Configurable Parameters**: Adjust temperature, top_p, and system prompts
- 🔒 **Secure Credentials**: Environment-based configuration for API keys

## Quick Start

### Using Ollama (Local)

```bash
# Install and start Ollama
brew install ollama
brew services start ollama
ollama pull mistral-nemo

# Run the app
pip install -r requirements.txt
streamlit run app.py
```

### Using watsonx (Cloud)

```bash
# Setup credentials
cp .env.example .env
# Edit .env and add your WATSONX_API_KEY and WATSONX_PROJECT_ID

# Run the app
pip install -r requirements.txt
streamlit run app.py
```

## Prerequisites

- [Docker](https://www.docker.com/) installed on your machine (optional)
- **For Ollama**: [Ollama](https://ollama.com/) running on your host machine
- **For watsonx**: IBM Cloud account with watsonx.ai access

## Installation

### Ollama Setup

To install Ollama using Homebrew:

```bash
brew install ollama
```

### watsonx Setup

1. **Get IBM Cloud API Key**:
   - Go to [IBM Cloud API Keys](https://cloud.ibm.com/iam/apikeys)
   - Create a new API key and save it securely

2. **Get watsonx Project ID**:
   - Go to your [watsonx project](https://dataplatform.cloud.ibm.com/wx/home)
   - Open your project settings
   - Copy the Project ID

3. **Configure Environment Variables**:
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env and add your credentials
   WATSONX_API_KEY=your-api-key-here
   WATSONX_PROJECT_ID=your-project-id-here
   WATSONX_URL=https://eu-gb.ml.cloud.ibm.com  # UK region (default)
   ```

## Managing Ollama

If you installed Ollama via Homebrew, you can manage the service with:

-   **Start Ollama**: `brew services start ollama`
-   **Stop Ollama**: `brew services stop ollama`

## Running Locally

1. Install dependencies:
   ```bash
   task install
   # or: pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   task run
   # or: streamlit run app.py
   ```

3. Open your browser and navigate to `http://localhost:8501`

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
-   **Install Model**: `task install:model`
-   **Run locally**: `task run`
-   **Run tests**: `task test`
-   **Build Docker**: `task docker:build`
-   **Run Docker**: `task docker:up`
-   **Stop Docker**: `task docker:down`

## Usage

### Provider Selection

In the sidebar, select your preferred provider:
- **Ollama (Local)**: Uses models running on your local machine
- **IBM watsonx**: Uses cloud-based watsonx.ai models

The app will automatically detect which providers are available based on:
- Ollama: Service running on localhost:11434
- watsonx: Valid credentials in `.env` file

### Chat Mode

- Upload context files (PDF, TXT, etc.)
- Enter your message in the chat input
- Use `@[path/to/file]` syntax to include file contents in prompts
- View streaming responses in real-time

### Text Transformation Mode

- Select a transformation template
- Enter or paste your text
- Click "Transform" to apply the template
- Copy or use the transformed output

## Configuration

### config.json

Configure default models and templates:

```json
{
  "default_provider": "ollama",
  "providers": {
    "ollama": {
      "default_model": "mistral-nemo:latest"
    },
    "watsonx": {
      "default_model": "meta-llama/llama-3-3-70b-instruct"
    }
  },
  "templates": {
    "Summarize": "Summarize the following text:",
    ...
  }
}
```

### Environment Variables

Create a `.env` file (see `.env.example` for template):

```bash
# watsonx Configuration
WATSONX_API_KEY=your-api-key-here
WATSONX_PROJECT_ID=your-project-id-here
WATSONX_URL=https://eu-gb.ml.cloud.ibm.com  # Optional, defaults to UK

# Ollama Configuration
OLLAMA_ENABLED=true                         # Set to 'false' to explicitly disable Ollama
```

### Optional Provider
The application will start gracefully even if a provider is missing or unreachable.
- **Ollama**: If the local service is not running, the Ollama provider will be marked as unavailable. You can also explicitly disable it by setting `OLLAMA_ENABLED=false`.
- **watsonx**: If credentials are not provided in the `.env` file, the watsonx provider will be marked as unavailable.

If no providers are available, the app will display a configuration guide.

## Development

The `docker-compose.yml` mounts the current directory to `/app` in the container, so changes to `app.py` will be reflected immediately (thanks to Streamlit's auto-reload).

### Running Tests

```bash
task test
# or: pytest tests/
```

## Troubleshooting

### Ollama Issues
- **"Could not connect to Ollama"**: Make sure Ollama is running (`brew services start ollama`)
- **"No models found"**: Pull a model first (`ollama pull mistral-nemo`)

### watsonx Issues
- **"watsonx credentials not configured"**: Check that `.env` file exists with valid credentials
- **"No providers available"**: Ensure either Ollama is running OR watsonx credentials are configured
- **Connection errors**: Verify your `WATSONX_URL` matches your IBM Cloud region


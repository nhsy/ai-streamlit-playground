# AI Streamlit Playground

![Build Status](https://github.com/nhsy/ai-streamlit-playground/actions/workflows/test.yml/badge.svg)

This is a Streamlit application that interfaces with local Ollama models and IBM watsonx.ai cloud models.

## Features

- 🔄 **Multi-Provider Support**: Switch between local Ollama, IBM watsonx, Google Gemini, and OpenRouter
- 💬 **Chat Interface**: Interactive chat with conversation history and file context
- 📥 **Model Management**: Download new models directly from the UI with progress tracking
- 📊 **Model Metadata**: Detailed tooltips showing size, parameter count, and quantization
- 🔄 **Text Transformation**: Apply templates for common text operations
- 📎 **File Upload**: Support for PDF, TXT, CSV, MD, PY, JSON, and more with `@[]` reference syntax
- 🎛️ **Configurable Parameters**: Adjust temperature, top_p, and system prompts
- 🔒 **Secure Credentials**: Environment-based configuration for API keys

## Quick Start

### Using Ollama (Local)

```bash
# Install and start Ollama
brew install ollama
brew services start ollama
ollama pull qwen2.5:7b

# Run the app
pip install -r requirements.txt
streamlit run app.py
```

### Using Cloud Providers (watsonx, Gemini, OpenRouter)

```bash
# Setup credentials
cp .env.example .env
# Edit .env and add API keys for your preferred providers:
# - WATSONX_API_KEY & WATSONX_PROJECT_ID
# - GEMINI_API_KEY
# - OPENROUTER_API_KEY

# Run the app
pip install -r requirements.txt
streamlit run app.py
```

## Prerequisites

- [Docker](https://www.docker.com/) installed on your machine (optional)
- **For Ollama**: [Ollama](https://ollama.com/) running on your host machine
- **For watsonx**: IBM Cloud account with watsonx.ai access
- **For Gemini**: Google Cloud project with Gemini API enabled
- **For OpenRouter**: OpenRouter account and API key

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

### Gemini Setup

1. **Get Gemini API Key**:
   - Go to [Google AI Studio](https://aistudio.google.com/)
   - Create a new API key

2. **Configure Environment Variables**:

   ```bash
   GEMINI_API_KEY=your-api-key-here
   ```

### OpenRouter Setup

1. **Get OpenRouter API Key**:
   - Go to [OpenRouter Keys](https://openrouter.ai/keys)
   - Create a key

2. **Configure Environment Variables**:

   ```bash
   OPENROUTER_API_KEY=your-api-key-here
   ```

## Managing Ollama

If you installed Ollama via Homebrew, you can manage the service with:

- **Start Ollama**: `brew services start ollama`
- **Stop Ollama**: `brew services stop ollama`

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

1. Make sure Ollama is running on your host machine (usually port 11434).
2. Build and run the container:

    ```bash
    docker-compose up --build
    ```

3. Open your browser and navigate to `http://localhost:8501`.

## Automation (Taskfile)

This project uses [Task](https://taskfile.dev/) to automate common commands.

- **Install dependencies**: `task install`
- **Install Model**: `task install:model`
- **Run locally**: `task run`
- **Run tests**: `task test`
- **Build Docker**: `task docker:build`
- **Run Docker**: `task docker:up`
- **Stop Docker**: `task docker:down`

## Usage

### Provider Selection

In the sidebar, select your preferred provider:

- **Ollama (Local)**: Uses models running on your local machine
- **IBM watsonx**: Uses cloud-based watsonx.ai models
- **Google Gemini**: Uses Google's Gemini models
- **OpenRouter**: Uses models aggregated by OpenRouter

The app will automatically detect which providers are available based on:

- Ollama: Service running on localhost:11434
- Cloud Providers: Valid API keys in `.env` file

### Model Selection & Management

- **Model Switcher**: Select from installed models. Hover over the info icon for details (size, params, quantization).
- **Download from Library**: Use the "Pull New Model" expander to:
  - Choose from suggested models optimized for your hardware (e.g., 16GB RAM).
  - Enter a custom model name from the Ollama library to download it.
  - Track download progress in real-time.

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

Configure default provider, models, and templates:

```json
{
  "default_provider": "gemini",
  "providers": {
    "ollama": {
      "default_model": "qwen2.5:7b"
    },
    "watsonx": {
      "default_model": "meta-llama/llama-3-3-70b-instruct"
    },
    "openrouter": {
      "default_model": "openrouter/auto",
      "models": {
         "openrouter/auto": "🤖 OpenRouter Auto (Free)",
         "meta-llama/llama-3.3-70b-instruct:free": "🆓 Llama 3.3 70B (Free)"
      }
    },
    "gemini": {
      "default_model": "gemini-2.5-flash"
    }
  },
  "templates": {
    "Summarize": "Summarize the following text:",
    "Extract Keywords": "Extract the main keywords from the following text:",
    "Fix Grammar": "Fix the grammar and spelling in the following text:",
    "Rewrite Professionally": "Rewrite the following text to sound more professional:",
    "Rewrite as a DevOps SME": "Rewrite the following text as a DevOps SME:"
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
WATSONX_ENABLED=true

# OpenRouter Configuration
OPENROUTER_API_KEY=your-api-key-here
OPENROUTER_ENABLED=true

# Google Gemini Configuration
GEMINI_API_KEY=your-api-key-here
GEMINI_ENABLED=true

# Ollama Configuration
OLLAMA_ENABLED=true                         # Set to 'false' to explicitly disable Ollama
```

### Optional Provider

The application will start gracefully even if a provider is missing or unreachable.

- **Ollama**: If the local service is not running, the Ollama provider will be marked as unavailable. You can also explicitly disable it by setting `OLLAMA_ENABLED=false`.
- **Cloud Providers**: If credentials (API keys) are not provided in the `.env` file, the respective provider (watsonx, Gemini, OpenRouter) will be marked as unavailable. Each can also be explicitly disabled using their `_ENABLED` flag.

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

### Cloud Provider Issues

- **"Credentials not configured"**: Check that `.env` file exists with valid keys for the provider you are trying to use.
- **"No providers available"**: Ensure either Ollama is running OR at least one cloud provider is configured.

### watsonx Specific

- **Connection errors**: Verify your `WATSONX_URL` matches your IBM Cloud region

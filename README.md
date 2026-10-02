# AI Streamlit Playground

![Build Status](https://github.com/nhsy/ai-streamlit-playground/actions/workflows/test.yml/badge.svg)

A Streamlit application for chatting with and transforming text using local Ollama models and cloud models from IBM watsonx.ai, Google Gemini and OpenRouter.

## Features

- 🔄 **Multi-Provider Support**: Switch between local Ollama, IBM watsonx, Google Gemini, and OpenRouter
- 💬 **Chat**: Streaming chat with conversation history; attached files stay in context for follow-up questions
- 📎 **File Attachments**: Attach PDF, DOCX, TXT, CSV, MD, PY, JSON and YAML files from the chat input, or reference local files with `@[path]`
- 📤 **Export**: Copy a chat as Markdown, or download it as Markdown or HTML
- 🔄 **Text Transformation**: Apply prompt templates to text, with streamed output
- 📥 **Model Management**: Download Ollama models from the UI with progress tracking
- 📊 **Model Metadata**: Tooltips showing size, parameter count and quantization
- 🎛️ **Configurable Parameters**: Adjust temperature, top_p and the system prompt
- 🎨 **Themed UI**: Light and dark themes in `.streamlit/config.toml`
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

The app has two pages, **Chat** and **Text Transformation**, listed at the top of the sidebar. The settings below them apply to both pages.

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
- **Download from Library**: Use the "Pull new model" expander to:
  - Choose from suggested models optimized for your hardware (e.g., 16GB RAM).
  - Enter a custom model name from the Ollama library to download it.
  - Track download progress in real-time.

### Chat

- Attach context files with the paperclip in the chat input
- Use `@[path/to/file]` to include a local file's contents in a prompt. Paths are relative to the project folder. Paths outside it and hidden files (such as `.env`) are refused, and files are truncated at 200 KB.
- Responses stream in real time. If a request fails, the error is shown and the message is not added to the history, so you can retry.
- Use **Export** to copy the chat or download it, and **Reset** to clear the chat and system prompt

### Text Transformation

- Select a transformation template
- Enter or paste your text
- Click "Transform" to stream the result

## Configuration

### config.json

Configure default provider, models, and templates:

```json
{
  "default_provider": "ollama",
  "providers": {
    "ollama": {
      "default_model": "gemma4"
    },
    "watsonx": {
      "default_model": "meta-llama/llama-3-3-70b-instruct"
    },
    "openrouter": {
      "default_model": "openrouter/auto",
      "models": {
         "openrouter/auto": "🤖 OpenRouter Auto (Free)",
         "meta-llama/llama-3.3-70b-instruct:free": "🆓 Llama 3.3 70B (Free)",
         "openai/gpt-oss-120b:free": "🆓 GPT-OSS 120B (Free)",
         "qwen/qwen3-next-80b-a3b-instruct:free": "🆓 Qwen3 Next 80B (Free)"
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

`default_provider` is a provider key: `ollama`, `watsonx`, `openrouter` or `gemini`. Text files in `templates/` are added as templates too; for example, `jira_story.txt` becomes "Jira Story".

### Theme

The light and dark themes are defined in `.streamlit/config.toml`. The app follows the browser's colour scheme.

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

Variables are read with `pydantic-settings` (`providers/settings.py`). The `*_ENABLED` flags accept `true`/`false`, `1`/`0` or `yes`/`no`; any other value is rejected with a validation error when the provider starts. API keys are held as secrets and are not shown in logs or reprs.

### Optional Provider

The application will start gracefully even if a provider is missing or unreachable.

- **Ollama**: If the local service is not running, the Ollama provider will be marked as unavailable. You can also explicitly disable it by setting `OLLAMA_ENABLED=false`.
- **Cloud Providers**: If credentials (API keys) are not provided in the `.env` file, the respective provider (watsonx, Gemini, OpenRouter) will be marked as unavailable. Each can also be explicitly disabled using their `_ENABLED` flag.

If no providers are available, the app will display a configuration guide.

## Development

The `docker-compose.yml` mounts the current directory to `/app` in the container, so code changes are picked up immediately by Streamlit's auto-reload.

### Project Layout

- `app.py`: entry point. It renders the sidebar settings and the page navigation.
- `views/chat.py`, `views/transform.py`: the two pages
- `core.py`: config, cached provider access, prompt expansion, file reading and chat export
- `models.py`: pydantic models for `config.json` (`AppConfig`), chat messages, stream chunks, model info and generation options
- `providers/`: one class per LLM provider, implementing `BaseProvider`
- `providers/settings.py`: per-provider environment settings (`pydantic-settings`)

Provider availability is cached for 30 seconds, and model lists for 60 seconds.

### Running Tests

```bash
task test
# or: PYTHONPATH=. uv run pytest --cov tests/
```

`task test` reports coverage and fails if total coverage drops below 80% (configured in `pyproject.toml`).

## Troubleshooting

### Ollama Issues

- **"Could not connect to Ollama"**: Make sure Ollama is running (`brew services start ollama`)
- **"No models found"**: Pull a model first (`ollama pull mistral-nemo`)

### Cloud Provider Issues

- **"Credentials not configured"**: Check that `.env` file exists with valid keys for the provider you are trying to use.
- **"No providers available"**: Ensure either Ollama is running OR at least one cloud provider is configured.

### watsonx Specific

- **Connection errors**: Verify your `WATSONX_URL` matches your IBM Cloud region

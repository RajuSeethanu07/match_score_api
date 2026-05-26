# Match Score API

Project structure:
match_score_api/
│
├── .venv/                      # Python local virtual environment
├── .env                        # Local file for cluster URIs and secret keys
├── requirements.txt            # Explicit third-party package dependencies pinned
│
├── config.py                   # Pydantic-settings boundary layer configuration
├── db_service.py               # Motor-async centralized MongoDB cluster database service
├── s3_parser.py                # Httpx file streaming & in-memory pypdf extraction pipeline
├── engine.py                   # 3-Step Hybrid Match Engine (Rules, Vectors, and LLM pass)
└── main.py                     # Central FastAPI entrypoint, routing, and concurrent orchestration

## Setup Instructions

1. Create and activate virtual environment:
   ```bash
   # Create project directory (if not already there)
   mkdir match_score_api
   cd match_score_api
   
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.template` to `.env` and fill in your values:
   ```bash
   copy .env.template .env  # Windows
   cp .env.template .env    # Linux/macOS
   ```

## Environment Variables

- `DEBUG`: Set to `True` for development, `False` for production
- `MONGO_CLUSTER_URI`: MongoDB connection string for the cluster
- `OPENAI_API_KEY`: API key for OpenAI services
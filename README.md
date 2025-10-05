# Universal RAG

## Project Structure
```
backend/
  app/
    api/
    core/
    models/
    services/
  data/
    bases/
      default/
        .gitkeep
requirements.txt
.env.example
.gitignore
```

## Getting Started
1. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
2. **Install dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment variables**: copy `.env.example` to `.env` and fill in the values.
   ```bash
   cp .env.example .env
   ```
4. **Run the FastAPI backend** (replace `app.main:app` with your ASGI application path when implemented):
   ```bash
   uvicorn app.main:app --reload --app-dir backend
   ```
5. **Run the Gradio interface** once implemented. For example, if you add an entry point in `backend/app/api/gradio_app.py`:
   ```bash
   python backend/app/api/gradio_app.py
   ```

## Notes
- Store your knowledge bases inside `backend/data/bases/`. A default location is prepared at `backend/data/bases/default/`.
- Use the packages under `backend/app/` to organize core logic, services, and API routes for the Retrieval-Augmented Generation system.

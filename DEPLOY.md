# Deployment Instructions (Vercel)

This project is configured for deployment on Vercel.

## Prerequisites

- Vercel Account
- Node.js & npm installed

## Steps

### Option 1: Deploy with Vercel CLI

1.  **Install Vercel CLI**:
    ```bash
    npm install -g vercel
    ```

2.  **Deploy**:
    Run the following command in the project root:
    ```bash
    vercel
    ```
    Follow the prompts.

3.  **Production Deployment**:
    ```bash
    vercel --prod
    ```

### Option 2: Deploy via GitHub (Recommended)

1.  Push this code to a GitHub repository.
2.  Log in to Vercel and click "Add New... > Project".
3.  Import your GitHub repository.
4.  Vercel should automatically detect the configuration.
    - **Build Command**: Leave empty (or default).
    - **Output Directory**: Leave empty.
    - **Install Command**: Vercel automatically installs from `requirements.txt`.

## Important Notes

- **AI Model Size**: This app uses `distilbert` (~250MB) and `torch` (~700MB). Vercel Serverless Functions have a limit of 50MB (zipped) / ~250MB (uncompressed).
- **Potential Failure**: The deployment **might fail** if the dependencies exceed the size limit or if the download takes too long.
- **Workaround**: If deployment fails, consider using:
    - **Render.com** (Docker support, no strict size limits).
    - **Railway.app**.
    - **Hugging Face Spaces**.

## Configuration

- `vercel.json`: Configures the Python runtime.
- `app.py`: contains logic to download NLTK data to `/tmp` and sets cache paths.

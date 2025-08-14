# From-Scratch AI Model: A Free, Self-Contained Language Model For Render

Welcome to the self-contained AI application! Unlike most AI chatbots, this one doesn't rely on external, paid AI services for its responses. It uses a unique language model built entirely from scratch right here in the app. This means you can get AI-powered conversations without any external inference costs.

***
## Key Features

* **Completely Local AI**: The chat bot is powered by an **Interpolated Kneser-Ney N-gram language model**. This model runs efficiently on a CPU, making it perfect for free-tier hosting services like Render.
* **Custom Training**: You can train the language model on your own data. Simply upload datasets (in JSON, JSONL, or ZIP formats) or even generate new data using Groq's API (used only for data generation, not for answering your questions).
* **Smart Answers with RAG**: The system uses **Retrieval-Augmented Generation (RAG)** to provide more accurate and contextually relevant answers. It fetches information from a vector database (**FAISS**) and uses that context to formulate a response.
* **Simple Setup**: Deploying this app is straightforward, especially on platforms like Render. The guide below provides all the necessary steps and environment variables.
* **Analytics and Administration**: The app includes a dedicated admin panel where you can view ratings, analytics, and other useful information about the model's performance.

***
## How It Works

The core of this application is a **word-level N-gram language model**. This model predicts the next word in a sequence based on the preceding words it has been trained on. 

The training process works like this:

1.  **Data Upload**: You provide the model with a dataset of text.
2.  **Training**: The app processes this data to build the N-gram model. This can be done via the **`/data`** page in the app or by calling the `POST /api/train/start` endpoint.
3.  **Generation**: When you chat with the bot, it uses this trained model to generate responses based on your input.

This approach is different from large-scale transformer models (like GPT-4), but it's a powerful and cost-effective way to create a functional AI chatbot.

***
## Deployment on Render

You can easily deploy this app on Render's free tier.

### Step 1: Push to GitHub

First, make sure all your code is in a GitHub repository, likely done by forking.

### Step 2: Configure Render

Create a new **Web Service** on Render and connect it to your GitHub repo.

* **Build Command**: `pip install -r requirements.txt`
* **Start Command**: `gunicorn -k uvicorn.workers.UvicornWorker app:app --timeout 180 --workers 1 --threads 8 --bind 0.0.0.0:$PORT`

### Step 3: Set Environment Variables

Add the following environment variables in your Render dashboard.

* `APP_ENV`: `production`
* `DB_URL`: `sqlite:///./data/app.db`
* `EMBED_MODEL_NAME`: `sentence-transformers/all-MiniLM-L6-v2`
* `RAG_TOP_K`: `8`
* `MAX_CONTEXT_CHARS`: `16000`
* **Optional**: `GROQ_API_KEY` (Only needed if you want to generate synthetic data with Groq)

***
## Get Started!

1.  Navigate to the **/data** page to upload your own JSON or JSONL datasets.
2.  Click the **"Train N-gram"** button to build the language model.
3.  Go to the chat page (`/`) and start a conversation with your newly trained AI!

***
## File Structure

* `app.py`: The main file that contains all the application logic, including the model, training, RAG, and user interface.
* `static/`: Contains all the front-end files for the chat, admin, and data pages.
* `data/`: This is where your uploaded files, tokenizer, model weights, and other important data are stored.

This project is open-source and released under the **MIT License**.

***
## Backups 

The app automatically creates local backups of your data in the `data/backups/` directory.

You can also set up **S3 backups** by adding these environment variables: `S3_BUCKET`, `S3_KEY`, `S3_SECRET`, and `S3_REGION`.

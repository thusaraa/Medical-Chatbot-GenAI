# Medical-Chatbot-GenAI

### STEP 1 - Create a conda environment after creating the hithub repository
```bash
conda create -n medibot python=3.10
```
```bash
conda activate medibot
```

### STEP 2 - Install requirement.txt
```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add Pinecone and OpenAI credentials as follows

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```bash
# Run the following command to store embeddings to Pinecone
python strore_index.py
```

```bash
# Finnaly run the following command
python app.py
```

Now,
```bash
open up localhost
```

### Techstack Used:

- Python
- Langchain
- Flask
- GPT
- Pinecone

# SCZ0BZq-jqY
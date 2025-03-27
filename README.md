# AI Tutor - Intelligent Question Answering System

## Overview
AI Tutor is an advanced question-answering system that utilizes multiple knowledge sources, including:

- **Vector Database (AstraDB):** Stores and retrieves indexed documents from uploaded PDFs and URLs.
- **Wikipedia:** Provides general knowledge and historical context.
- **arXiv:** Fetches the latest research papers and scientific advancements.
- **LLM-based Reasoning (DeepSeek and Groq LLMs):** Generates answers requiring logical synthesis and deep understanding.

The system intelligently routes user queries to the appropriate source using a structured routing model.

## Features
- Upload and vectorize PDFs for document-based Q&A.
- Retrieve information from structured vector stores.
- Search Wikipedia for general knowledge queries.
- Fetch scientific papers from arXiv.
- Generate responses using advanced LLMs when external sources are insufficient.
- Streamlit-based chat interface for interactive Q&A.

## Installation & Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.9+
- `pip`

### Clone the Repository
```sh
git clone https://github.com/your-repo/ai-tutor.git
cd ai-tutor
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root and add the necessary API keys:
```
ASTRA_DB_APPLICATION_TOKEN=your_astra_db_token
ASTRA_DB_ID=your_astra_db_id
LANGCHAIN_API_KEY=your_langchain_api_key
GROQ_API_KEY=your_groq_api_key
```

### Run the Application
```sh
streamlit run app.py
```

## How to Use

1. **Upload PDFs**: Click the upload section and select multiple PDF files.
2. **Enter URLs**: Input website links for document retrieval.
3. **Ask Questions**: Use the chat input box to ask any question.
4. **Processing**: The system routes the query and provides an accurate answer.

## Technologies Used
- **LangChain**: For LLM integration and retrieval-based Q&A.
- **Cassandra & AstraDB**: For storing and retrieving vectorized documents.
- **Streamlit**: For building an interactive web-based UI.
- **HuggingFace Embeddings**: To vectorize documents for efficient retrieval.
- **Wikipedia & arXiv APIs**: For external knowledge sources.
- **DeepSeek & Groq LLMs**: For natural language understanding and generation.

## License
This project is licensed under the MIT License.

## Contributions
Feel free to contribute by submitting pull requests or reporting issues.

## Contact
For any questions or suggestions, open an issue or contact the developer.


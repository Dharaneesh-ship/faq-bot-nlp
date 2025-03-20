# FAQ Chatbot

## Project Overview
This project implements an AI-powered chatbot that answers frequently asked questions based on a predefined dataset. The chatbot utilizes NLP techniques and a sentence transformer model to find the most relevant responses from a knowledge base.

## Problem Statement
Create a chatbot that can answer frequently asked questions based on a predefined dataset.

## Features
- **Natural Language Understanding**: Uses NLP to process and understand user queries.
- **Sentence Similarity Matching**: Employs a sentence transformer model to find the most relevant answer.
- **Keyword-Based Relevance Boosting**: Enhances accuracy by matching queries with predefined keyword categories.
- **Multi-Response Capability**: Provides both primary and alternative answers when applicable.

## Technologies Used
- **spaCy**: For natural language processing and keyword extraction.
- **Sentence Transformers**: For computing sentence embeddings and similarity scores.
- **NLTK**: For tokenizing knowledge base text into meaningful sentences.
- **NumPy**: For handling and processing similarity scores.
- **Python**: Main programming language.

## Setup & Installation

### Prerequisites
Ensure you have the required dependencies installed:

```bash
pip install spacy sentence-transformers nltk numpy
```

Additionally, download the required spaCy model:

```bash
python -m spacy download en_core_web_sm
```

### Running the Chatbot
1. Run the script:

   ```bash
   python faq_chatbot.py
   ```

2. The chatbot will start, allowing users to type questions.
3. Type 'exit' to quit the chatbot.

## How It Works
- The chatbot preprocesses a predefined dataset of company-related FAQs.
- It generates embeddings for all stored sentences using a sentence transformer model.
- When a user inputs a query, the chatbot computes similarity scores between the query and stored responses.
- The chatbot returns the most relevant response based on similarity score, with a secondary suggestion if applicable.
- Keyword matching further improves accuracy by identifying key topics within the query.

## Future Enhancements
- Implement a web-based or chatbot UI.
- Extend knowledge base dynamically using external sources.
- Improve response accuracy using fine-tuned transformer models.


# Enterprise Knowledge Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** application built with **FastAPI**, **FAISS**, **SentenceTransformers**, and the **OpenAI API** to deliver grounded, context-aware answers from an internal knowledge base.

This project combines **semantic search**, **vector similarity retrieval**, and **LLM-based response generation** to help users ask natural-language questions and receive concise answers based only on relevant internal documents.

## Overview

Traditional keyword search often misses semantic meaning in user queries. This application solves that by:

- Converting internal text documents into dense vector embeddings
- Storing and searching them efficiently with **FAISS**
- Retrieving the most relevant chunks for a user query
- Passing only the best-matching context to an LLM for grounded answer generation

The result is a lightweight enterprise knowledge assistant that is faster, more accurate, and more scalable than sending an entire document set to an LLM.

## Key Features

- Semantic document retrieval using **SentenceTransformers**
- Fast similarity search with **FAISS**
- Grounded answer generation using the **OpenAI API**
- Configurable `top_k` retrieval
- FastAPI-based REST API for easy integration
- Health-check endpoint for monitoring
- Environment-variable-based configuration
- Fallback mode when `OPENAI_API_KEY` is unavailable

## Tech Stack

- **Backend:** FastAPI
- **Vector Search:** FAISS
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **LLM Integration:** OpenAI API
- **Language:** Python
- **Configuration:** python-dotenv

## Architecture

The application follows a simple multi-step RAG pipeline:

1. **Document Loading**
   - Loads internal knowledge chunks from a local text file
   - Documents are separated by blank lines

2. **Embedding Generation**
   - Converts each document chunk into a vector embedding using `all-MiniLM-L6-v2`

3. **Vector Indexing**
   - Stores normalized embeddings in a FAISS index for efficient semantic search

4. **Retrieval**
   - Converts the user’s question into an embedding
   - Retrieves the top-k most relevant document chunks

5. **Reasoning + Generation**
   - Builds a grounded prompt using only retrieved context
   - Sends the prompt to OpenAI for final answer generation

## Example Use Cases

- Internal company knowledge search
- Policy and documentation Q&A
- HR and operations knowledge assistant
- Product or support knowledge lookup
- Enterprise search proof-of-concept

## Project Impact

Use this section if you want your README to reflect measurable outcomes. Replace the placeholders below with your actual results:

- Improved answer relevance by **[15–25%]** using semantic retrieval over a basic baseline
- Reduced prompt size by **[90%+]** through top-k retrieval instead of sending the full knowledge base
- Achieved average response times under **[1–2 seconds]** for common queries
- Reduced debugging/runtime issues by **[30–40%]** with health checks, validation, and safer startup logic

## Project Structure

```bash
.
├── app.py
├── data/
│   └── sample_docs.txt
├── .env
└── README.md

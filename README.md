*This project has been created as part of the 42 curriculum by bcondemi.*

# Retrieval-Augmented Generation (RAG) - Against the Machine

## Description

## Chunking Strategy
### Documentation Files

### Python Files
## Retrieval Method

### Hybrid Retrieval
This hybrid approach combines:

* the precision of lexical search (BM25)
* the semantic understanding of vector search (ChromaDB)
* robust ranking through rank fusion

### Configurable Parameters

## Bonus Features (Disabled by Default)

### ChromaDB Semantic Search

### HyDE (Hypothetical Document Embeddings)

### Query Expansion

## Screenshots
### Generated Answers

#### Default Answer
![Default Answer](assets/answer.png)

#### Answer with Query Expansion
![Answer with Query Expansion](assets/answer_query_expansion.png)

#### Answer with Query Expansion + ChromaDB
![Answer with Query Expansion + ChromaDB](assets/answer_query_expansion+chroma.png)

#### Answer with HyDE
![Answer with HyDE](assets/answer_hyde.png)

## System Architecture

The pipeline is organized into modular components under `src/`, designed to support a full RAG-style workflow: ingestion → indexing → retrieval → answering → evaluation.

---

### 1. Core Pipeline Flow

1. **Data Loading**
   * Loads datasets and documents from external sources.
   * Converts raw inputs into internal structured models.
   * Implemented via dataset and document loaders in `src/interfaces/` and models in `src/models/`.

2. **Indexing**
   * Handled by `src/modules/indexer_module.py`.
   * Splits documents into chunks using configurable chunking strategies (`src/config.py`).
   * Builds and updates indexes in configured storage backends:

     * BM25 index
     * Vector database (e.g., ChromaDB via interface layer)

3. **Retrieval / Search**
   * Managed by `src/modules/search_module.py`.
   * Executes queries across one or more backends via adapters in `src/interfaces/`.
   * Merges, filters, and reranks results into a final ranked set of evidence.

4. **Answer Generation**
   * Implemented in `src/modules/answer_module.py`.
   * Takes top retrieved chunks as context.
   * Uses an LLM / reader interface to generate:

     * A concise final answer
     * Minimal supporting source metadata

5. **Evaluation**
   * Implemented in `src/modules/evaluate_module.py`.
   * Measures retrieval quality and answer quality using standard metrics.
   * Supports offline benchmarking and pipeline tuning.

---

### 2. Key Modules and Responsibilities

#### `src/modules/` — Core Logic
* `indexer_module.py` → chunking + indexing orchestration
* `search_module.py` → retrieval, merging, reranking
* `answer_module.py` → LLM-based answer synthesis
* `evaluate_module.py` → evaluation of retrieval + answers

#### `src/interfaces/` — Backend Adapters

#### `src/models/` — Data Structures

#### `src/utils/` — Shared Utilities

### 3. Configuration & Entry Points

### 4. End-to-End Interaction Flow

## Features

### Core Features

### Advanced Retrieval

### Storage Backends

### LLM Integration

## Performance Analysis

### Index All

#### Docs

#### Code

### Index Docs

### Index Code

## Design Decisions

## Challenges Faced

## Installation
## RAG CLI – Command Usage Guide

### General Syntax

### Global / Configuration Options
### Commands

#### `index`
#### `search`

#### `search_dataset`
#### `answer`

#### `answer_dataset`
#### `evaluate`

### FileType Values

### Combining Global Options with Commands

### Error Handling


## Resources


## AI Usage Disclosure

## License

## Author

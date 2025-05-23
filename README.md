[![License: MIT][mit-shield]](./LICENSE)

# Job Recommender AI
This repository contains a hybrid job recommendation system that uses both user interaction and job posting data to generate relevant job suggestions.

## Overview
- Build a job-job graph from user interactions
- Generate job embeddings using both graph-based and text-based features
- Combine embeddings and builds a vector database for fast similarity search
- Recommend top-N similar jobs fora a given job ID

## Dataset
- **User Events**: User event information including clicks and applications on job listings.
- **Items**: Item information including titles and descriptions of job listings.

> Note: For confidentiality reasons, only the first 5 rows of each dataset have been retained as examples.

## Setup
Install required libraries:

> cd <your_directory>\Job-Recommender-AI

> pip install -r requirements.txt

## Usage

### Graph Construction and Analysis
To build a job-job graph from user interactions and analyze its structure:

> python build_graph.py

### Graph-Based Embeddings
To generate job embeddings using Graph Convolutional Networks (GCN) based on the graph:

> python generate_graph_embeds.py

### Text-Based Embeddings
To generate job embeddings using Sentence-BERT (SBERT) based on job titles and descriptions:

> python generate_nlp_embeds.py

### Vector Database
To combine graph and text embeddings; to build a FAISS vector database for fast similarity search:

> python build_faiss_db.py

### Job Recommendation
To recommend top-N similar jobs for a given job ID:

> python recommend_job.py <job_id> top_n <recommendation_count>

## Additional Information
- The complete contents (around 400 MB) of the `results` folder can be accessed through [this link](https://drive.google.com/drive/folders/1sMs2LKjlKbzhQJwgsMxxYo3-e0dvupFX?usp=sharing), simply replacing the `results` folder is enough.

## License
© 2025 [Nural Ozel](https://github.com/tfb-sv).

This work is licensed under the [MIT License](./LICENSE).

[cc-by]: https://creativecommons.org/licenses/by/4.0/
[mit-shield]: https://img.shields.io/badge/License-MIT-yellow.svg

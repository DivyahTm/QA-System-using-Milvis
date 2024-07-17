# NVIDIA CUDA Documentation QA System

This project implements a Question Answering system for NVIDIA CUDA documentation using web crawling, vector databases, and natural language processing techniques.

## Setup Instructions

1. Clone this repository:
    git clone https://github.com/DivyahTm/nvidia-cuda-qa-system.git
    cd nvidia-cuda-qa-system
    
2. Create a virtual environment and activate it:
    python -m venv venv
    venv\Scripts\activate

3. Install the required packages:
    pip install -r requirements.txt

4. Set up Milvus:
    - Follow the Milvus installation guide: https://milvus.io/docs/install_standalone-docker.md
    - Start the Milvus server
    (Simple Steps Install Docker and run the following command in your terminal :-
    wget https://github.com/milvus-io/milvus/releases/download/v2.3.4/milvus-standalone-docker-compose.yml -O docker-compose.yml)

5. Set your Hugging Face API token in the code.

## Running the System

1. Start the Streamlit app:
    streamlit run app.py
    OR python -m streamlit run app.py

2. Enter your question about NVIDIA CUDA in the text input field and click "Get Answer"

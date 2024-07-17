import os
import torch
import scrapy
import streamlit as st
from scrapy import signals
from bs4 import BeautifulSoup
from pymilvus import connections, utility
from scrapy.crawler import CrawlerProcess
from langchain.llms import HuggingFaceHub
from scrapy.signalmanager import dispatcher
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Milvus
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer, util
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up environment variables
os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'hf_.....'  # Replace with your Hugging Face API token

# Milvus connection settings
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'nvidia_cuda_docs'

st.title("NVIDIA CUDA Documentation QA System")

# Spider for crawling NVIDIA CUDA documentation
class NvidiaCudaSpider(scrapy.Spider):
    name = 'nvidia_cuda'
    allowed_domains = ['docs.nvidia.com']
    start_urls = ['https://docs.nvidia.com/cuda/']
    max_depth = 5

    def parse(self, response):
        depth = response.meta.get('depth', 0)
        if depth < self.max_depth:
            soup = BeautifulSoup(response.body, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            yield {
                'url': response.url,
                'text': text
            }
            
            for link in response.css('a::attr(href)'):
                yield response.follow(link, callback=self.parse, meta={'depth': depth + 1})

@st.cache_resource
def run_spider():
    results = []
    
    def crawler_results(signal, sender, item, response, spider):
        results.append(item)
    
    dispatcher.connect(crawler_results, signal=signals.item_passed)
    
    process = CrawlerProcess(settings={
        'LOG_LEVEL': 'ERROR',
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    })
    
    process.crawl(NvidiaCudaSpider)
    process.start()
    
    return results

@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_documents():
    crawled_data = run_spider()
    if not crawled_data:
        st.warning("No documents were retrieved during web crawling. Please check the website accessibility and crawling settings.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = []
    for item in crawled_data:
        chunks = text_splitter.split_text(item['text'])
        docs.extend([{'chunk': chunk, 'metadata': {'url': item['url']}} for chunk in chunks])
    return docs

@st.cache_resource
def initialize_milvus():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = load_documents()
    
    if not docs:
        st.error("No documents to index. The vector database could not be initialized.")
        return None, []
    
    texts = [doc['chunk'] for doc in docs]
    metadatas = [doc['metadata'] for doc in docs]
    
    vectorstore = Milvus.from_texts(
        texts,
        embeddings,
        collection_name=COLLECTION_NAME,
        metadatas=metadatas,
        index_params={
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        }
    )
    return vectorstore, texts

vectorstore, texts = initialize_milvus()

if vectorstore is None:
    st.error("The QA system could not be initialized due to lack of data. Please check the crawling settings and try again.")
else:
    model = load_sentence_transformer()
    bm25_retriever = BM25Retriever.from_texts(texts)
    milvus_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    def rerank(query, documents, top_k=4):
        query_embedding = model.encode(query, convert_to_tensor=True)
        document_embeddings = model.encode([doc.page_content for doc in documents], convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
        top_results = torch.topk(scores, k=min(top_k, len(documents)))
        return [documents[i] for i in top_results.indices]

    def custom_hybrid_rerank(inputs):
        milvus_docs = inputs["milvus_retrieved_doc"]
        bm25_docs = inputs["bm25_retrieved_doc"]
        query = inputs["query"]
        
        # Combine documents
        combined_docs = milvus_docs + bm25_docs
        
        # Remove duplicates based on content
        seen = set()
        unique_docs = []
        for doc in combined_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
        
        return rerank(query, unique_docs)

    hybrid_retriever = RunnableParallel(
        milvus_retrieved_doc=milvus_retriever,
        bm25_retrieved_doc=bm25_retriever,
        query=RunnablePassthrough()
    )

    # Initialize the Hugging Face LLM
    llm = HuggingFaceHub(
    repo_id="openai-community/gpt2-large", 
    model_kwargs={"temperature": 0.3, "max_length": 100}    
)
    rag_prompt_template = PromptTemplate.from_template("""Answer the following question based on the given context:

    Context: {context}

    Question: {question}

    Answer:""")

    # Now define your chain
    hybrid_and_rerank_chain = (
        {
            "context": hybrid_retriever | custom_hybrid_rerank,
            "question": RunnablePassthrough(),
        }
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )

    # Streamlit UI
    query = st.text_input("Enter your question about NVIDIA CUDA:")

    if st.button("Get Answer"):
        if query:
            with st.spinner("Processing..."):
                result = hybrid_and_rerank_chain.invoke(query)

            st.subheader("Answer:")
            st.write(result)

            st.subheader("Sources:")
            retrieved_docs = custom_hybrid_rerank({"milvus_retrieved_doc": milvus_retriever.get_relevant_documents(query), 
                                                   "bm25_retrieved_doc": bm25_retriever.get_relevant_documents(query), 
                                                   "query": query})
            for i, doc in enumerate(retrieved_docs, 1):
                st.write(f"{i}. [Source]({doc.metadata['url']})")
        else:
            st.warning("Please enter a question.")

st.sidebar.title("About")
st.sidebar.info(
    "This QA System uses NVIDIA CUDA documentation, LangChain, Milvus (HNSW), "
    "BM25, and SentenceTransformers for hybrid search and reranking."
)

from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.ollama.document_embedder import OllamaDocumentEmbedder
from haystack_integrations.components.embedders.ollama.text_embedder import OllamaTextEmbedder

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

documents = [
    Document(content="I saw a black horse running"),
    Document(content="Germany has many big cities"),
    Document(content="My name is Wolfgang and I live in Berlin"),
]

document_embedder = OllamaDocumentEmbedder()
documents_with_embeddings = document_embedder.run(documents)["documents"]
document_store.write_documents(documents_with_embeddings)

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", OllamaTextEmbedder())
query_pipeline.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

# query = "Who lives in Berlin?"
query = "who sees a black horse running?"

result = query_pipeline.run({"text_embedder": {"text": query}})

print(result["retriever"]["documents"][0])
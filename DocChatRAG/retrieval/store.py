import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from django.conf import settings


class VectorStore:
    def __init__(self):
        # HuggingFace Endpoint Embeddings using API
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-mpnet-base-v2",  # or your preferred embedding model
            task="feature-extraction",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        self.collection_name = "docchat"
        self._vector_store = None

    def get_vector_store(self):
        """Initialize and return Chroma vector store"""
        if self._vector_store is None:
            if self._is_chroma_cloud():
                # Chroma Cloud setup
                self._vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
                    tenant=os.getenv("CHROMA_TENANT"),
                    database=os.getenv("CHROMA_DATABASE")
                )
            else:
                # Local Chroma setup for development
                persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
                self._vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory
                )
        return self._vector_store

    def _is_chroma_cloud(self):
        """Check if Chroma Cloud credentials are available"""
        return all([
            os.getenv("CHROMA_API_KEY"),
            os.getenv("CHROMA_TENANT"),
            os.getenv("CHROMA_DATABASE")
        ])

    def get_retriever(self, search_kwargs=None):
        """Get retriever from vector store"""
        if search_kwargs is None:
            search_kwargs = {"k": 4}

        vector_store = self.get_vector_store()
        return vector_store.as_retriever(search_kwargs=search_kwargs)


# Global instance
vector_store_manager = VectorStore()

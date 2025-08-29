import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    TextLoader,
    WebBaseLoader,
    UnstructuredImageLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from retrieval.store import vector_store_manager

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Complete document ingestion pipeline for RAG"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_store = vector_store_manager.get_vector_store()

    def load_document(self, source: str, source_type: str = None) -> List[Document]:
        """Load documents from various sources"""
        try:
            if source_type is None:
                source_type = self._detect_source_type(source)

            loader = self._get_loader(source, source_type)
            documents = loader.load()

            # Add source metadata
            for doc in documents:
                doc.metadata.update({
                    'source': source,
                    'source_type': source_type,
                    'ingestion_timestamp': self._get_timestamp()
                })

            logger.info(f"Loaded {len(documents)} documents from {source}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load document from {source}: {e}")
            return []

    def _detect_source_type(self, source: str) -> str:
        """Auto-detect source type based on file extension or URL"""
        if source.startswith(('http://', 'https://')):
            return 'web'

        path = Path(source)
        extension = path.suffix.lower()

        type_mapping = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',
            '.csv': 'csv',
            '.txt': 'text',
            '.md': 'text',
            '.py': 'text',
            '.json': 'text',
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image'
        }

        return type_mapping.get(extension, 'text')

    def _get_loader(self, source: str, source_type: str):
        """Get appropriate document loader based on source type"""
        loaders = {
            'pdf': lambda: PyPDFLoader(source),
            'docx': lambda: Docx2txtLoader(source),
            'csv': lambda: CSVLoader(
                file_path=source,
                csv_args={
                    'delimiter': ',',
                    'quotechar': '"',
                    'fieldnames': None  # Auto-detect headers
                }
            ),
            'text': lambda: TextLoader(source, encoding='utf-8'),
            'web': lambda: WebBaseLoader(
                web_paths=[source],
                bs_kwargs={
                    'features': 'html.parser'
                }
            ),
            'image': lambda: UnstructuredImageLoader(source)
        }

        if source_type not in loaders:
            raise ValueError(f"Unsupported source type: {source_type}")

        return loaders[source_type]()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks with metadata preservation"""
        all_chunks = []

        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])

            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': f"{doc.metadata.get('source', 'unknown')}_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'original_source': doc.metadata.get('source')
                })
                all_chunks.append(chunk)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def store_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Store document chunks in vector store"""
        if not documents:
            return {'success': False, 'message': 'No documents to store'}

        try:
            # Generate unique IDs for each chunk
            ids = [doc.metadata.get('chunk_id', f"chunk_{i}")
                   for i, doc in enumerate(documents)]

            # Add documents to vector store
            self.vector_store.add_documents(documents=documents, ids=ids)

            result = {
                'success': True,
                'message': f'Successfully stored {len(documents)} document chunks',
                'chunk_count': len(documents),
                'sources': list(set(doc.metadata.get('original_source')
                                    for doc in documents))
            }

            logger.info(result['message'])
            return result

        except Exception as e:
            error_msg = f"Failed to store documents: {e}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg}

    def process_source(self, source: str, source_type: str = None) -> Dict[str, Any]:
        """Complete pipeline: load -> split -> store"""
        try:
            # Load documents
            documents = self.load_document(source, source_type)
            if not documents:
                return {'success': False, 'message': 'No documents loaded'}

            # Split into chunks
            chunks = self.split_documents(documents)
            if not chunks:
                return {'success': False, 'message': 'No chunks created'}

            # Store in vector database
            result = self.store_documents(chunks)

            # Add processing summary
            result['processing_summary'] = {
                'source': source,
                'source_type': source_type or self._detect_source_type(source),
                'original_documents': len(documents),
                'total_chunks': len(chunks)
            }

            return result

        except Exception as e:
            error_msg = f"Pipeline failed for {source}: {e}"
            logger.error(error_msg)
            return {'success': False, 'message': error_msg}

    def process_multiple_sources(self, sources: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process multiple sources in batch"""
        results = []
        total_chunks = 0
        successful_sources = 0

        for source_config in sources:
            source = source_config.get('source')
            source_type = source_config.get('type')

            result = self.process_source(source, source_type)
            results.append({
                'source': source,
                'result': result
            })

            if result['success']:
                successful_sources += 1
                total_chunks += result.get('chunk_count', 0)

        summary = {
            'success': successful_sources > 0,
            'total_sources': len(sources),
            'successful_sources': successful_sources,
            'failed_sources': len(sources) - successful_sources,
            'total_chunks_stored': total_chunks,
            'detailed_results': results
        }

        logger.info(f"Batch processing complete: {successful_sources}/{len(sources)} sources processed")
        return summary

    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata"""
        from datetime import datetime
        return datetime.now().isoformat()


# Global instance
ingestion_pipeline = DocumentIngestionPipeline()

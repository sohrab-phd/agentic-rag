import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


class VectorDbManager:
    __client: QdrantClient
    __dense_embeddings: HuggingFaceEmbeddings
    __sparse_embeddings: FastEmbedSparse

    def __init__(self):
        self.__client = QdrantClient(path=config.QDRANT_DB_PATH)
        self.__dense_embeddings = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)
        self.__sparse_embeddings = FastEmbedSparse(
            model_name=config.SPARSE_MODEL,
            disable_stemmer=config.BM25_DISABLE_STEMMER,
        )

    def _dense_vector_size(self) -> int:
        return len(self.__dense_embeddings.embed_query("آزمایش"))

    def _dense_vector_params(self, collection_name: str) -> qmodels.VectorParams | None:
        if not self.__client.collection_exists(collection_name):
            return None
        info = self.__client.get_collection(collection_name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            for key in ("", "dense", "text-dense"):
                if key in vectors and hasattr(vectors[key], "size"):
                    return vectors[key]
            for value in vectors.values():
                if hasattr(value, "size"):
                    return value
            return None
        return vectors

    def _existing_collection_vector_size(self, collection_name: str) -> int | None:
        params = self._dense_vector_params(collection_name)
        return params.size if params is not None else None

    def _sample_stored_dense_dimension(self, collection_name: str) -> int | None:
        if not self.__client.collection_exists(collection_name):
            return None
        if self.__client.count(collection_name).count == 0:
            return None
        points, _ = self.__client.scroll(
            collection_name=collection_name,
            limit=1,
            with_vectors=True,
        )
        if not points:
            return None
        vec = points[0].vector
        if vec is None:
            return None
        if isinstance(vec, dict):
            sparse_name = config.SPARSE_VECTOR_NAME
            for key in ("", "dense", "text-dense"):
                if key in vec and isinstance(vec[key], list):
                    return len(vec[key])
            for key, value in vec.items():
                if key != sparse_name and isinstance(value, list):
                    return len(value)
            return None
        if isinstance(vec, list):
            return len(vec)
        return None

    def _collection_needs_recreate(self, collection_name: str, expected_size: int) -> bool:
        if not self.__client.collection_exists(collection_name):
            return False
        schema_size = self._existing_collection_vector_size(collection_name)
        stored_size = self._sample_stored_dense_dimension(collection_name)
        if schema_size is not None and schema_size != expected_size:
            return True
        if stored_size is not None and stored_size != expected_size:
            return True
        return False

    def create_collection(self, collection_name) -> bool:
        """Create or validate collection. Returns True if vectors were wiped/recreated."""
        expected_size = self._dense_vector_size()
        recreated = False

        if self._collection_needs_recreate(collection_name, expected_size):
            schema_size = self._existing_collection_vector_size(collection_name)
            stored_size = self._sample_stored_dense_dimension(collection_name)
            print(
                f"WARNING: Collection '{collection_name}' is incompatible with "
                f"embeddings ({expected_size}-dim). "
                f"schema={schema_size}, stored_points={stored_size}. "
                f"Recreating collection - re-indexing documents."
            )
            self.delete_collection(collection_name)
            recreated = True

        if self.__client.collection_exists(collection_name):
            print(f"[OK] Collection already exists: {collection_name}")
            return recreated

        print(f"Creating collection: {collection_name}...")
        self.__client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=expected_size, distance=qmodels.Distance.COSINE),
            sparse_vectors_config={config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()},
        )
        print(f"[OK] Collection created: {collection_name}")
        return True

    def should_reindex_markdown(self, collection_name: str, collection_was_recreated: bool) -> bool:
        from pathlib import Path

        md_files = list(Path(config.MARKDOWN_DIR).glob("*.md"))
        if not md_files:
            return False
        if not self.__client.collection_exists(collection_name):
            return True
        if collection_was_recreated:
            return True
        if self.__client.count(collection_name).count == 0:
            return True
        stored_size = self._sample_stored_dense_dimension(collection_name)
        expected_size = self._dense_vector_size()
        return stored_size is not None and stored_size != expected_size

    def delete_collection(self, collection_name):
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Removing existing Qdrant collection: {collection_name}")
                self.__client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: could not delete collection {collection_name}: {e}")

    def get_collection(self, collection_name) -> QdrantVectorStore:
        try:
            return QdrantVectorStore(
                client=self.__client,
                collection_name=collection_name,
                embedding=self.__dense_embeddings,
                sparse_embedding=self.__sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID,
                sparse_vector_name=config.SPARSE_VECTOR_NAME,
            )
        except Exception as e:
            print(f"Unable to get collection {collection_name}: {e}")
            raise

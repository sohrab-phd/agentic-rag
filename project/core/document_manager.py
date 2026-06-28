from pathlib import Path
import shutil
import config
import locale_fa as L
from utils import pdfs_to_markdowns, clear_directory_contents

class DocumentManager:

    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.markdown_dir = Path(config.MARKDOWN_DIR)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        
    def add_documents(self, document_paths, progress_callback=None):
        if not document_paths:
            return 0, 0
            
        document_paths = [document_paths] if isinstance(document_paths, str) else document_paths
        document_paths = [p for p in document_paths if p and Path(p).suffix.lower() in [".pdf", ".md"]]
        
        if not document_paths:
            return 0, 0
            
        added = 0
        skipped = 0
            
        for i, doc_path in enumerate(document_paths):
            if progress_callback:
                progress_callback((i + 1) / len(document_paths), L.UI_PROCESSING.format(name=Path(doc_path).name))
                
            doc_name = Path(doc_path).stem
            md_path = self.markdown_dir / f"{doc_name}.md"
            
            if md_path.exists():
                skipped += 1
                continue
                
            try:            
                if Path(doc_path).suffix.lower() == ".md":
                    shutil.copy(doc_path, md_path)
                else:
                    pdfs_to_markdowns(str(doc_path), overwrite=False)            
                parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)
                
                if not child_chunks:
                    skipped += 1
                    continue
                
                collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)
                collection.add_documents(child_chunks)
                self.rag_system.parent_store.save_many(parent_chunks)
                
                added += 1
                
            except Exception as e:
                print(f"Error processing {doc_path}: {e}")
                skipped += 1
            
        return added, skipped
    
    def get_markdown_files(self):
        if not self.markdown_dir.exists():
            return []
        return sorted([p.name.replace(".md", ".pdf") for p in self.markdown_dir.glob("*.md")])
    
    def _reset_vector_store(self):
        if not self.rag_system.vector_db.delete_collection(self.rag_system.collection_name):
            raise RuntimeError(
                "Could not clear the vector database. "
                "Stop other running instances of this app and try again."
            )
        self.rag_system.vector_db.create_collection(self.rag_system.collection_name)

    def reindex_all(self, progress_callback=None) -> int:
        """Re-chunk and embed every markdown file (e.g. after embedding model change)."""
        md_files = sorted(self.markdown_dir.glob("*.md"))
        if not md_files:
            return 0

        self.rag_system.parent_store.clear_store()
        self._reset_vector_store()
        collection = self.rag_system.vector_db.get_collection(self.rag_system.collection_name)

        indexed = 0
        for i, md_path in enumerate(md_files):
            if progress_callback:
                progress_callback((i + 1) / len(md_files), L.UI_PROCESSING.format(name=md_path.name))
            try:
                parent_chunks, child_chunks = self.rag_system.chunker.create_chunks_single(md_path)
                if not child_chunks:
                    continue
                collection.add_documents(child_chunks)
                self.rag_system.parent_store.save_many(parent_chunks)
                indexed += 1
            except Exception as e:
                print(f"Error reindexing {md_path}: {e}")
        return indexed

    def clear_all(self):
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        clear_directory_contents(self.markdown_dir)
        
        self.rag_system.parent_store.clear_store()
        self._reset_vector_store()

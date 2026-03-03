import hashlib
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

from research_analyst.config import settings
from research_analyst.vectorstore.store import get_vectorstore

console = Console()


def _doc_hash(content: str) -> str:
    """Generate a short hash to detect duplicate chunks."""
    return hashlib.md5(content.encode()).hexdigest()[:12]


def ingest_documents(
    data_dir: str,
    company_name: str,
    ticker: str,
    doc_type: str = "report",
) -> int:
    """
    Load documents from data_dir, chunk them, enrich with metadata,
    and store in ChromaDB. Returns the number of chunks ingested.

    Skips chunks already present in the collection (deduplication via hash).
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        console.print(f"[red]Directory not found: {data_dir}[/red]")
        return 0

    # --- Load documents ---
    raw_docs = []

    pdf_loader = DirectoryLoader(
        str(data_path), glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True
    )
    txt_loader = DirectoryLoader(
        str(data_path), glob="**/*.txt", loader_cls=TextLoader, silent_errors=True
    )

    raw_docs.extend(pdf_loader.load())
    raw_docs.extend(txt_loader.load())

    if not raw_docs:
        console.print(f"[yellow]No PDF or TXT files found in {data_dir}[/yellow]")
        return 0

    console.print(f"[green]Loaded {len(raw_docs)} document(s)[/green]")

    # --- Chunk documents ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    console.print(f"[green]Split into {len(chunks)} chunk(s)[/green]")

    # --- Enrich metadata and deduplicate ---
    vectorstore = get_vectorstore()
    existing_ids = set(vectorstore.get()["ids"])

    new_chunks = []
    new_ids = []

    for chunk in chunks:
        chunk_id = _doc_hash(chunk.page_content)
        if chunk_id in existing_ids:
            continue
        chunk.metadata.update(
            {
                "company": company_name,
                "ticker": ticker,
                "doc_type": doc_type,
            }
        )
        new_chunks.append(chunk)
        new_ids.append(chunk_id)

    if not new_chunks:
        console.print("[yellow]All chunks already ingested — nothing new to add.[/yellow]")
        return 0

    # --- Embed and store ---
    vectorstore.add_documents(new_chunks, ids=new_ids)
    console.print(
        f"[bold green]Ingested {len(new_chunks)} new chunk(s) into "
        f"collection '{settings.collection_name}'[/bold green]"
    )
    return len(new_chunks)

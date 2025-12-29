from pdf2image import convert_from_path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import base64
from io import BytesIO
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from pathlib import Path
from datetime import datetime, timezone
import uuid
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance
import logging
import time

load_dotenv()

def get_pdf_logger(pdf_name: str):
    """
    Creates (or returns) a dedicated logger for a single PDF.
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logger_name = f"pdf_logger_{pdf_name}"

    if logger_name in logging.Logger.manager.loggerDict:
        existing_logger = logging.getLogger(logger_name)
        if existing_logger.handlers:
            return existing_logger

    pdf_logger = logging.getLogger(logger_name)
    pdf_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # File handler (per PDF)
    file_handler = logging.FileHandler(
        logs_dir / f"{pdf_name}.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    pdf_logger.addHandler(file_handler)
    pdf_logger.addHandler(console_handler)

    # prevent duplicate logs
    pdf_logger.propagate = False

    return pdf_logger


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),               # console
        logging.FileHandler("ingestion_global.log")   # file
    ]
)

global_logger  = logging.getLogger(__name__)

def extract_page_content(llm, image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Extract ALL content from this scanned page.\n"
                    "- Preserve headings\n"
                    "- Convert tables to Markdown tables\n"
                    "- Write equations in LaTeX-style text\n"
                    "- Describe diagrams briefly if present\n"
                    "- Do NOT hallucinate\n"
                    "- Output clean markdown"
                )
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            }
        ]
    )

    response = llm.invoke([message])

    return response.content

def save_documents_to_json(documents, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

def is_pdf_path(pdf_path: str) -> bool:
    path = Path(pdf_path)
    return path.is_file() and path.suffix.lower() == ".pdf"

def extract_content_from_pages(llm, pages, logger):
    logger.info(f"Starting OCR for {len(pages)} pages")
    documents = []

    for i, page in enumerate(pages):
        try:
            text = extract_page_content(llm, page)
            documents.append({
                "page": i + 1,
                "content": text
            })
            logger.info(f"OCR completed for page {i + 1}")
            time.sleep(1)
        except Exception as e:
            logger.exception(f"OCR failed for page {i + 1}")
            time.sleep(1)
            continue

    return documents

def convert_to_langchain_document(documents):
    lc_docs = [
        Document(
            page_content=d["content"],
            metadata={"page": d["page"]}
        )
        for d in documents
    ]
    return lc_docs

def split_documents(lc_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=300
    )

    chunks = splitter.split_documents(lc_docs)
    return chunks

def get_chunks_with_enriched_metadata(chunks, pdf_path, subject, semester, unit, embedding_model, version):
    raw_chunks = []
    for chunk in chunks:
        raw_chunks.append(
            {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
        )
    
    pdf_name = Path(pdf_path).name
    document_id = Path(pdf_path).stem.lower().replace(" ", "_")
    ingestion_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"

    for i, chunk in enumerate(raw_chunks):
        chunk["metadata"]["semester"] = semester
        chunk["metadata"]["subject"] = subject
        chunk["metadata"]["unit"] = unit
        chunk["metadata"]["chunk_id"] = f"{document_id}_p{chunk['metadata']['page']}_c{i + 1}"
        chunk["metadata"]["document_id"] = document_id
        chunk["metadata"]["ingestion_id"] = ingestion_id
        chunk["metadata"]["source_pdf"] = pdf_name
        chunk["metadata"]["embedding_model"] = embedding_model
        chunk["metadata"]["ingestion_version"] = version
        

    enriched_metadata_chunks = []

    for chunk in raw_chunks:
        enriched_metadata_chunks.append(Document(
            page_content = chunk["page_content"],
            metadata = chunk["metadata"]
        ))
    
    return enriched_metadata_chunks

def save_chunks(chunks, chunk_cache_file_path):
    chunk_list = []
    for chunk in chunks:
        chunk_list.append(
            {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
        )

    with open(chunk_cache_file_path, "w", encoding="utf-8") as file:
        json.dump(chunk_list, file, ensure_ascii=False, indent=2)

def get_existing_embedding_model(client, collection_name):
    """
    Returns the embedding_model used in the collection (from metadata),
    or None if the collection is empty.
    """
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=False
    )

    if not points:
        return None  # collection exists but is empty

    payload = points[0].payload
    metadata = payload.get("metadata", {})  # get metadata dict
    return metadata.get("embedding_model")


def add_to_vector_db(chunks, client, embedding_model, collection_name, embedding_size, logger):

    # Case 1: Collection does NOT exist → create it
    if not client.collection_exists(collection_name):
        logger.info(f"Creating new Qdrant collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_size,
                distance=Distance.COSINE
            )
        )

    # Case 2: Collection exists → verify embedding model
    else:
        existing_embedding_model = get_existing_embedding_model(
            client, collection_name
        )

        if existing_embedding_model is not None and existing_embedding_model != embedding_model:
            raise ValueError(
                f"Embedding model mismatch!\n"
                f"Collection '{collection_name}' already uses "
                f"'{existing_embedding_model}', but you are trying to add "
                f"embeddings from '{embedding_model}'."
            )

    # Safe to proceed
    logger.info(f"Using existing Qdrant collection: {collection_name}")
    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        embedding=OpenAIEmbeddings(model=embedding_model),
        client=client
    )

    vector_store.add_documents(chunks)

def load_chunk_ingest_pdf(pdf_path, subject, semester, unit, version, vision_model, client, collection_name, embedding_model_name, embedding_size):
    pdf_name = Path(pdf_path).stem.replace(" ", "_")
    pdf_logger = get_pdf_logger(pdf_name)
    global_logger.info(f"PDF started: {pdf_name}")
    
    pdf_logger.info(
        f"Starting ingestion | "
        f"pdf={pdf_name}, "
        f"semester={semester}, "
        f"subject={subject}, "
        f"unit={unit}, "
        f"version={version}"
    )
    
    if not is_pdf_path(pdf_path):
        pdf_logger.warning(f"Invalid file type: {pdf_path}")
        global_logger.warning(f"Invalid file type: {pdf_path}")
        return
    
    Path("extracted_documents").mkdir(parents=True, exist_ok=True)
    Path("extracted_chunks").mkdir(parents=True, exist_ok=True)
    
    pdf_logger.info(f"Starting PDF page extraction: {pdf_name}")
    pages = convert_from_path(pdf_path, dpi=300)
    
    if not pages:
        pdf_logger.warning(f"No pages extracted from PDF: {pdf_name}")
        global_logger.warning(f"No pages extracted from PDF: {pdf_name}")
        return
    
    pdf_logger.info(f"PDF page extraction successful | pages={len(pages)}")
    print()
    
    documents = extract_content_from_pages(vision_model, pages, logger = pdf_logger)
    pdf_logger.info(f"Content extraction from pages successful! Total documents created: {len(documents)}")
    print()
    
    document_id = Path(pdf_path).stem.lower().replace(" ", "_")
    ocr_cache_file_path = (
        Path("extracted_documents")
        / f"{document_id}_ocr_cache_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    )
    
    pdf_logger.info(f"Saving extracted documents to {ocr_cache_file_path}...")
    save_documents_to_json(documents, ocr_cache_file_path)
    pdf_logger.info(f"Extracted documents saved to {ocr_cache_file_path} successfully!")
    print()
    
    pdf_logger.info("Converting extracted documents to langchain document format...")
    lc_docs = convert_to_langchain_document(documents)
    pdf_logger.info("Documents converted successfully to the langchain document format!")
    print()
    
    pdf_logger.info("Splitting the extracted documents into chunks...")
    chunks = split_documents(lc_docs)
    pdf_logger.info(f"Splitting Successful! Total chunks created: {len(chunks)}")
    print()
    
    pdf_logger.info("Enriching the metadata of all the chunks...")
    enriched_metadata_chunks = get_chunks_with_enriched_metadata(chunks = chunks, pdf_path = pdf_path, subject = subject, semester = semester, unit = unit, embedding_model = embedding_model_name, version = version)
    pdf_logger.info("Metadata enrichment of chunks successful!")
    print()
    
    chunk_cache_file_path = (
        Path("extracted_chunks")
        / f"{document_id}_chunks_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    )
    
    pdf_logger.info(f"Saving chunks with enriched metadata to {chunk_cache_file_path}...")
    save_chunks(enriched_metadata_chunks, chunk_cache_file_path)
    pdf_logger.info(f"Chunks with enriched metadata successfully saved to {chunk_cache_file_path}!")
    
    pdf_logger.info(
        f"Starting vector DB ingestion | "
        f"collection={collection_name}, "
        f"embedding_model={embedding_model_name}, "
        f"chunks={len(enriched_metadata_chunks)}, "
        f"pdf={pdf_name}, "
        f"semester={semester}, "
        f"subject={subject}, "
        f"unit={unit}, "
        f"version={version}"
    )
    
    try:
        add_to_vector_db(
            chunks=enriched_metadata_chunks,
            client=client,
            embedding_model=embedding_model_name,
            collection_name=collection_name,
            embedding_size=embedding_size,
            logger = pdf_logger
        )
    except Exception as e:
        pdf_logger.exception("Failed during vector DB ingestion")
        global_logger.exception(f"PDF failed: {pdf_name}")
        raise
    else:
        pdf_logger.info("Vector DB ingestion successful!")

        global_logger.info(f"PDF completed: {pdf_name}")

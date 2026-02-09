import logging

from langchain_experimental.text_splitter import SemanticChunker

from text_to_json.clients import get_embeddings

logger = logging.getLogger(__name__)

# Default parameters for the fallback RecursiveCharacterTextSplitter.
DEFAULT_FALLBACK_CHUNK_SIZE: int = 8000
DEFAULT_FALLBACK_CHUNK_OVERLAP: int = 400


def semantic_chunk(
    text: str,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 95.0,
    min_chunk_size: int = 500,
) -> list[str]:
    """
    Divide the text into semantically coherent chunks.

    Uses embeddings to identify natural breakpoints where the meaning changes significantly.

    Args:
        text: The text to be divided into chunks.
        breakpoint_threshold_type: Type of threshold for break.
            - "percentile": Uses percentile of distances (default: 95)
            - "standard_deviation": Uses standard deviation
            - "interquartile": Uses interquartile range
            - "gradient": Uses gradient of distances
        breakpoint_threshold_amount: Value of threshold (meaning depends on the type).
        min_chunk_size: Minimum size of each chunk in characters.

    Returns:
        List of strings, each being a semantically coherent chunk.
    """
    if not text or not text.strip():
        return []

    chunker = SemanticChunker(
        embeddings=get_embeddings(),
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )

    documents = chunker.create_documents([text])

    chunks = [doc.page_content for doc in documents]

    filtered_chunks = _merge_small_chunks(chunks, min_chunk_size)

    return filtered_chunks


def _merge_small_chunks(chunks: list[str], min_size: int) -> list[str]:
    """Merge small chunks with the previous one to avoid excessive fragmentation."""
    if not chunks:
        return []

    result = []
    buffer = ""

    for chunk in chunks:
        if buffer:
            combined = buffer + "\n\n" + chunk
            if len(buffer) < min_size:
                buffer = combined
            else:
                result.append(buffer)
                buffer = chunk
        else:
            buffer = chunk

    if buffer:
        if result and len(buffer) < min_size:
            result[-1] = result[-1] + "\n\n" + buffer
        else:
            result.append(buffer)

    return result


def chunk_with_fallback(
    text: str,
    chunk_size: int = DEFAULT_FALLBACK_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_FALLBACK_CHUNK_OVERLAP,
) -> list[str]:
    """
    Try semantic chunking, with fallback to RecursiveCharacterTextSplitter.

    Args:
        text: The text to be divided.
        chunk_size: Target size of each chunk (used in fallback).
        chunk_overlap: Overlap between chunks (used in fallback).

    Returns:
        List of chunks.
    """
    try:
        return semantic_chunk(text)
    except Exception as e:  # noqa: BLE001 — intentionally broad
        # Semantic chunking can fail for many reasons (embedding API errors,
        # empty texts, numerical issues in breakpoint detection, etc.).
        # The except is intentionally broad because *any* failure should
        # trigger the deterministic fallback so extraction can proceed.
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        logger.warning("Semantic chunking failed (%s: %s). Using fallback recursive.", type(e).__name__, e)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        return splitter.split_text(text)

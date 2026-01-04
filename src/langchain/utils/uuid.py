import uuid
import hashlib
from typing import List

NAMESPACE = uuid.NAMESPACE_DNS


def generate_stable_uuid_for_text(text: str) -> str:
    """Generates a deterministic UUID v5 from a text string or image path(Path().name) or pdf_path_page_idx."""
    text_bytes = text.encode("utf-8")
    content_hash = hashlib.sha256(text_bytes).hexdigest()
    stable_uuid = uuid.uuid5(NAMESPACE, content_hash)
    return str(stable_uuid)

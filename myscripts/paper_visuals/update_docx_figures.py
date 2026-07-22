"""Replace selected embedded figures in a thesis DOCX without rebuilding the document.

The image relationship is resolved from the drawing immediately preceding each
caption. Only the referenced media bytes are replaced, so document styles,
equations, captions, pagination settings, and unrelated media remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def sha256(data: bytes) -> str:
    """Return the SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def paragraph_text(paragraph: ET.Element) -> str:
    """Return visible text from one Word paragraph."""
    return "".join(node.text or "" for node in paragraph.findall(".//w:t", NS)).strip()


def relationship_target_for_caption(entries: dict[str, bytes], caption_prefix: str) -> tuple[str, str]:
    """Resolve the embedded media target immediately preceding a caption."""
    document = ET.fromstring(entries["word/document.xml"])
    paragraphs = document.findall(".//w:body/w:p", NS)
    caption_index = next(
        (index for index, paragraph in enumerate(paragraphs) if paragraph_text(paragraph).startswith(caption_prefix)),
        None,
    )
    if caption_index is None:
        raise ValueError(f"Caption starting with {caption_prefix!r} was not found")

    embed_id: str | None = None
    for paragraph in reversed(paragraphs[:caption_index]):
        blip = paragraph.find(".//a:blip", NS)
        if blip is not None:
            embed_id = blip.get(f"{{{NS['r']}}}embed")
            if embed_id:
                break
    if not embed_id:
        raise ValueError(f"No embedded image was found before caption {caption_prefix!r}")

    relationships = ET.fromstring(entries["word/_rels/document.xml.rels"])
    relationship = next(
        (node for node in relationships.findall("rel:Relationship", NS) if node.get("Id") == embed_id),
        None,
    )
    if relationship is None:
        raise ValueError(f"Relationship {embed_id!r} was not found")

    target = relationship.get("Target")
    if not target:
        raise ValueError(f"Relationship {embed_id!r} has no target")
    normalized = str(PurePosixPath("word") / PurePosixPath(target))
    if not normalized.startswith("word/media/") or normalized not in entries:
        raise ValueError(f"Unexpected or missing media target: {normalized}")
    return embed_id, normalized


def parse_replacement(value: str) -> tuple[str, Path]:
    """Parse a ``caption-prefix=image.png`` replacement specification."""
    prefix, separator, image = value.partition("=")
    if not separator or not prefix.strip() or not image.strip():
        raise argparse.ArgumentTypeError("Use CAPTION_PREFIX=IMAGE_PATH, for example 图1=figure1.png")
    path = Path(image).resolve()
    if path.suffix.lower() != ".png":
        raise argparse.ArgumentTypeError("Replacement images must be PNG files")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Replacement image does not exist: {path}")
    return prefix.strip(), path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="append",
        type=parse_replacement,
        required=True,
        metavar="CAPTION_PREFIX=IMAGE_PATH",
        help="May be supplied more than once.",
    )
    args = parser.parse_args()

    docx_path = args.docx.resolve()
    with zipfile.ZipFile(docx_path, "r") as source:
        infos = source.infolist()
        entries = {info.filename: source.read(info.filename) for info in infos}

    original_hashes = {name: sha256(data) for name, data in entries.items()}
    replacements: dict[str, tuple[str, Path, bytes]] = {}
    for prefix, image_path in args.replace:
        relationship_id, target = relationship_target_for_caption(entries, prefix)
        replacement = image_path.read_bytes()
        if target in replacements:
            raise ValueError(f"Multiple captions resolve to the same media target: {target}")
        if not target.lower().endswith(".png"):
            raise ValueError(f"{prefix!r} currently targets {target}, not a PNG")
        entries[target] = replacement
        replacements[target] = (relationship_id, image_path, replacement)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{docx_path.stem}_", suffix=".docx", dir=docx_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary_path, "w") as destination:
            for info in infos:
                destination.writestr(info, entries[info.filename])
        os.replace(temporary_path, docx_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    with zipfile.ZipFile(docx_path, "r") as verified:
        verified_entries = {info.filename: verified.read(info.filename) for info in verified.infolist()}

    if set(verified_entries) != set(original_hashes):
        raise RuntimeError("The DOCX package entry set changed unexpectedly")
    for name, before_hash in original_hashes.items():
        if name not in replacements and sha256(verified_entries[name]) != before_hash:
            raise RuntimeError(f"Unrelated DOCX entry changed unexpectedly: {name}")

    for target, (relationship_id, image_path, replacement) in replacements.items():
        after_hash = sha256(verified_entries[target])
        if after_hash != sha256(replacement):
            raise RuntimeError(f"Replacement did not persist: {target}")
        print(f"{relationship_id} -> {target} <- {image_path}")
        print(f"SHA-256: {after_hash}")
    print(f"Updated DOCX: {docx_path}")


if __name__ == "__main__":
    main()

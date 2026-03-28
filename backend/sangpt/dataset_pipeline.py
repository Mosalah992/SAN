"""
Dataset manifest and ingestion pipeline for SANCTA-GPT.
Normalizes raw project datasets into JSONL corpora and records source metadata.
"""

import csv
import json
import hashlib
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Iterable, Any, Tuple


@dataclass
class DatasetEntry:
    dataset_id: str
    category: str
    source_path: str
    processed_path: str
    format: str
    examples: int
    chars: int
    sha256: str
    updated_at: str


class DatasetManifestPipeline:
    """Scans raw data, writes normalized JSONL corpora, and persists a manifest."""

    def __init__(self, data_root: str = "./DATA"):
        self.data_root = Path(data_root)
        self.processed_root = self.data_root / "processed"
        self.manifests_root = self.data_root / "manifests"
        self.processed_root.mkdir(parents=True, exist_ok=True)
        self.manifests_root.mkdir(parents=True, exist_ok=True)

    def build_manifest(self) -> Dict[str, Any]:
        dataset_entries: List[DatasetEntry] = []
        grouped_records: Dict[str, List[Dict[str, Any]]] = {
            "knowledge": [],
            "security": [],
            "convo": [],
            "all": [],
        }

        for category, file_path in self._discover_sources():
            records = self._ingest_file(file_path, category)
            processed_path = self.processed_root / f"{file_path.stem}.jsonl"
            self._write_jsonl(processed_path, records)
            dataset_entries.append(
                DatasetEntry(
                    dataset_id=file_path.stem.lower().replace(" ", "_"),
                    category=category,
                    source_path=str(file_path),
                    processed_path=str(processed_path),
                    format=file_path.suffix.lower().lstrip("."),
                    examples=len(records),
                    chars=sum(len(record["text"]) for record in records),
                    sha256=self._sha256(file_path),
                    updated_at=datetime.now().isoformat(),
                )
            )
            grouped_records.setdefault(category, []).extend(records)
            grouped_records["all"].extend(records)

        grouped_records["conversational"] = list(grouped_records["convo"])

        # Build a balanced corpus: oversample convo to ~50%, downsample security/knowledge
        balanced = self._build_balanced_corpus(grouped_records)
        grouped_records["balanced"] = balanced

        corpus_paths = {}
        for mode, records in grouped_records.items():
            output_path = self.processed_root / f"corpus_{mode}.jsonl"
            self._write_jsonl(output_path, records)
            corpus_paths[mode] = str(output_path)

        manifest = {
            "generated_at": datetime.now().isoformat(),
            "data_root": str(self.data_root),
            "datasets": [asdict(entry) for entry in dataset_entries],
            "corpora": corpus_paths,
        }
        manifest_path = self.manifests_root / "dataset_manifest.json"
        manifest["manifest_path"] = str(manifest_path)

        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        return manifest

    def load_manifest(self) -> Dict[str, Any]:
        manifest_path = self.manifests_root / "dataset_manifest.json"
        if not manifest_path.exists():
            return self.build_manifest()
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def ensure_manifest(self) -> Dict[str, Any]:
        return self.build_manifest()

    def load_training_documents(self, mode: str = "all") -> List[str]:
        manifest = self.load_manifest()
        normalized_mode = (mode or "all").strip().lower()
        if normalized_mode == "conversation":
            normalized_mode = "convo"
        corpus_path = Path(manifest["corpora"].get(normalized_mode, manifest["corpora"]["all"]))
        documents = []
        with corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text", "").strip()
                if text:
                    documents.append(text)
        return documents

    def summarize_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "generated_at": manifest.get("generated_at"),
            "datasets": len(manifest.get("datasets", [])),
            "examples": sum(entry.get("examples", 0) for entry in manifest.get("datasets", [])),
            "corpora": manifest.get("corpora", {}),
        }

    def _deduplicate_records(self, records: List[Dict[str, Any]], similarity_prefix: int = 120) -> List[Dict[str, Any]]:
        """Remove near-duplicate records by comparing a prefix of the answer text.

        Templated datasets produce rows where the answer differs only in a few
        slot-filled words.  Hashing a truncated prefix of the answer collapses
        those near-duplicates while keeping genuinely distinct entries.
        """
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for record in records:
            text = record.get("text", "")
            # Use the response portion if structured as USER:/ASSISTANT:
            answer = text
            if "\nASSISTANT: " in text:
                answer = text.split("\nASSISTANT: ", 1)[1]
            # Normalize: lowercase, collapse whitespace, take prefix
            norm = " ".join(answer.lower().split())[:similarity_prefix]
            if norm not in seen:
                seen.add(norm)
                unique.append(record)
        return unique

    def _build_balanced_corpus(self, grouped: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Build a balanced corpus: convo ~20%, security ~40%, knowledge ~40%.

        Deduplicates convo records first to remove template noise, then
        lightly oversamples convo so the model learns conversational patterns
        without drowning out domain knowledge.
        """
        convo = self._deduplicate_records(list(grouped.get("convo", [])))
        security = list(grouped.get("security", []))
        knowledge = list(grouped.get("knowledge", []))

        if not convo:
            return security + knowledge

        # Target: convo is ~20% of the total corpus
        domain_total = len(security) + len(knowledge)
        target_convo = max(len(convo), domain_total // 4)

        # Oversample convo by repeating (only if needed)
        if len(convo) < target_convo:
            balanced_convo = convo * (target_convo // max(len(convo), 1))
            remainder = target_convo - len(balanced_convo)
            if remainder > 0:
                balanced_convo.extend(random.sample(convo, min(remainder, len(convo))))
        else:
            balanced_convo = convo[:target_convo]

        result = balanced_convo + security + knowledge
        random.shuffle(result)
        return result

    def _discover_sources(self) -> Iterable[Tuple[str, Path]]:
        """Auto-discover all CSV and TXT datasets in DATA subdirectories."""
        # Category mapping by directory
        dir_category = {
            "knowledge": "knowledge",
            "security": "security",
            "conversational": "convo",
        }
        seen = set()
        for subdir_name, category in dir_category.items():
            subdir = self.data_root / subdir_name
            if not subdir.exists():
                continue
            for ext in ("*.csv", "*.txt"):
                for file_path in sorted(subdir.glob(ext)):
                    if file_path.stat().st_size < 10:
                        continue  # Skip empty files
                    real = file_path.resolve()
                    if real in seen:
                        continue
                    seen.add(real)
                    yield category, file_path

    def _ingest_file(self, file_path: Path, category: str) -> List[Dict[str, Any]]:
        if file_path.suffix.lower() == ".csv":
            return self._ingest_csv(file_path, category)
        if file_path.suffix.lower() == ".txt":
            return self._ingest_text(file_path, category)
        return []

    def _ingest_csv(self, file_path: Path, category: str) -> List[Dict[str, Any]]:
        records = []
        with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)

        if not rows:
            return records

        header = [cell.strip().lower() for cell in rows[0]]
        question_idx = header.index("question") if "question" in header else None
        answer_idx = header.index("answer") if "answer" in header else None
        start_idx = 1 if question_idx is not None else 0

        for offset, row in enumerate(rows[start_idx:], start=1):
            values = [cell.strip() for cell in row if cell and cell.strip()]
            if not values:
                continue
            if question_idx is not None and answer_idx is not None and len(row) > max(question_idx, answer_idx):
                prompt = row[question_idx].strip()
                response = row[answer_idx].strip()
                text = f"USER: {prompt}\nASSISTANT: {response}"
            else:
                prompt = None
                response = None
                text = " | ".join(values)
            records.append(
                {
                    "record_id": f"{file_path.stem}:{offset}",
                    "dataset_id": file_path.stem.lower().replace(" ", "_"),
                    "category": category,
                    "source_path": str(file_path),
                    "text": text,
                    "prompt": prompt,
                    "response": response,
                    "meta": {"row_number": offset},
                }
            )
        return records

    def _ingest_text(self, file_path: Path, category: str) -> List[Dict[str, Any]]:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
        records = []
        for idx, chunk in enumerate(chunks, start=1):
            records.append(
                {
                    "record_id": f"{file_path.stem}:{idx}",
                    "dataset_id": file_path.stem.lower().replace(" ", "_"),
                    "category": category,
                    "source_path": str(file_path),
                    "text": chunk,
                    "prompt": None,
                    "response": None,
                    "meta": {"chunk_number": idx},
                }
            )
        return records

    def _write_jsonl(self, output_path: Path, records: List[Dict[str, Any]]):
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

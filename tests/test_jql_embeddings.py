import re
import pytest
import numpy as np
from pathlib import Path

_ANNOTATION_FILE = Path(__file__).parent.parent / "data" / "jira_jql_annotated_queries.md"


def test_annotation_file_pair_count_matches_last_annotation_number():
    from rag.jql_embeddings import JQL_Embeddings

    pairs = JQL_Embeddings._parse_jql_annotations(None, _ANNOTATION_FILE)

    text = _ANNOTATION_FILE.read_text(encoding="utf-8")
    annotation_blocks = re.findall(r'/\*\s*\d+\.', text)
    expected = len(annotation_blocks)

    print(f"\nParsed {len(pairs)} pairs, annotation blocks in file: {expected}")
    assert len(pairs) == expected, (
        f"Parsed {len(pairs)} pairs but file contains {expected} annotation blocks — "
        f"{expected - len(pairs)} entries were merged or skipped"
    )

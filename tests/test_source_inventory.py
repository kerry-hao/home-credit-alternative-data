"""In-memory unit tests for the Task 06 source-inventory helpers."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "inventory_sources.py"
SPEC = importlib.util.spec_from_file_location("inventory_sources", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


@pytest.mark.parametrize(
    ("file_name", "group", "depth", "shard", "status"),
    [
        ("train_static_0.parquet", "train_static_0", 0, None, "EXPECTED_PATTERN"),
        ("train_applprev_1_0.parquet", "train_applprev_1", 1, 0, "EXPECTED_PATTERN"),
        ("train_credit_bureau_a_2_10.parquet", "train_credit_bureau_a_2", 2, 10, "EXPECTED_PATTERN"),
        ("train_base.parquet", "train_base", None, None, "BASE"),
        ("odd.parquet", "odd", None, None, "UNEXPECTED_PATTERN"),
    ],
)
def test_provisional_filename_grouping(file_name, group, depth, shard, status):
    result = inventory.provisional_filename_group(file_name)
    assert result["provisional_table_group"] == group
    assert result["inferred_depth"] == depth
    assert result["inferred_shard"] == shard
    assert result["naming_status"] == status


def test_dictionary_matching_statuses_and_empty_description():
    rows, mapping, findings = inventory.validate_dictionary_rows(
        [
            {"Variable": "income_1A", "Description": "Income, quoted \"gross\""},
            {"Variable": "empty_2A", "Description": ""},
        ]
    )
    assert rows[0]["Description"] == "Income, quoted \"gross\""
    assert findings["empty_description_variable_names"] == ["empty_2A"]
    assert inventory.dictionary_status("income_1A", mapping) == (
        "MATCHED",
        "Income, quoted \"gross\"",
    )
    assert inventory.dictionary_status("empty_2A", mapping) == (
        "MATCHED_EMPTY_DESCRIPTION",
        "",
    )
    assert inventory.dictionary_status("case_id", mapping) == (
        "STRUCTURAL_NOT_IN_DICTIONARY",
        "",
    )
    assert inventory.dictionary_status("mystery_3A", mapping) == ("UNDOCUMENTED", "")


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"Variable": "income_1A", "Description": "one"},
            {"Variable": "income_1A", "Description": "two"},
        ],
        [{"Variable": "  ", "Description": "blank"}],
    ],
)
def test_dictionary_ambiguous_names_are_rejected(rows):
    with pytest.raises(inventory.DictionaryValidationError):
        inventory.validate_dictionary_rows(rows)


def test_proposal_candidate_extraction_exact_boundaries_and_locations():
    blocks = [
        {
            "location": "paragraph:1",
            "text": "Use income_1A, score_940, and unknown_77Z; ignore xincome_1A_more.",
        },
        {
            "location": "table:1/row:1/cell:1/paragraph:1",
            "text": "income_1A and case_id are referenced here.",
        },
        {"location": "paragraph:2", "text": "income_1A appears again."},
    ]
    extracted = inventory.extract_proposal_candidates(
        blocks,
        ["income_1A", "score_940"],
        ["case_id"],
    )
    assert sorted(extracted) == ["case_id", "income_1A", "score_940", "unknown_77Z"]
    assert extracted["income_1A"]["locations"] == [
        "paragraph:1",
        "table:1/row:1/cell:1/paragraph:1",
        "paragraph:2",
    ]
    assert "xincome_1A_more" not in extracted
    assert len(extracted["income_1A"]["contexts"]) <= 3


def _field(file_name, group, name, arrow_type):
    return {
        "file_name": file_name,
        "provisional_table_group": group,
        "field_name": name,
        "arrow_type": arrow_type,
    }


def test_candidate_location_mapping_and_deterministic_ordering():
    extracted = {
        "absent_1A": {"locations": ["paragraph:3"], "contexts": ["absent_1A"]},
        "multi_2A": {"locations": ["paragraph:2"], "contexts": ["multi_2A"]},
        "sharded_3A": {"locations": ["paragraph:1"], "contexts": ["sharded_3A"]},
    }
    fields = [
        _field("train_z_1_1.parquet", "train_z_1", "sharded_3A", "int64"),
        _field("train_z_1_0.parquet", "train_z_1", "sharded_3A", "int32"),
        _field("train_b_0.parquet", "train_b_0", "multi_2A", "double"),
        _field("train_a_0.parquet", "train_a_0", "multi_2A", "float"),
    ]
    rows = inventory.build_candidate_inventory(extracted, {}, fields)
    assert [row["candidate_name"] for row in rows] == [
        "absent_1A",
        "multi_2A",
        "sharded_3A",
    ]
    by_name = {row["candidate_name"]: row for row in rows}
    assert by_name["absent_1A"]["location_status"] == "NOT_OBSERVED_IN_TRAIN"
    assert by_name["sharded_3A"]["location_status"] == "OBSERVED_IN_ONE_GROUP"
    assert by_name["sharded_3A"]["physical_files"] == [
        "train_z_1_0.parquet",
        "train_z_1_1.parquet",
    ]
    assert by_name["sharded_3A"]["arrow_types"] == ["int32", "int64"]
    assert by_name["multi_2A"]["location_status"] == "OBSERVED_IN_MULTIPLE_GROUPS"
    assert by_name["multi_2A"]["provisional_groups"] == ["train_a_0", "train_b_0"]


def test_internal_report_counts_reconcile():
    tables = [{"column_count": 2}, {"column_count": 1}]
    fields = [{}, {}, {}]
    candidates = [
        {
            "candidate_name": "a_1A",
            "location_status": "OBSERVED_IN_ONE_GROUP",
            "observed_in_train": True,
            "physical_file_count": 2,
            "provisional_group_count": 1,
        },
        {
            "candidate_name": "b_2A",
            "location_status": "NOT_OBSERVED_IN_TRAIN",
            "observed_in_train": False,
            "physical_file_count": 0,
            "provisional_group_count": 0,
        },
    ]
    checks = inventory.build_internal_checks(tables, fields, candidates)
    assert checks["field_occurrence_count_reconciles"] is True
    assert checks["candidate_status_counts_reconcile"] is True
    assert checks["all_internal_checks_pass"] is True


def test_csv_and_json_round_trip_commas_and_quotes():
    value = 'Description with, comma and "quotes"'
    buffer = io.StringIO()
    inventory.write_csv_rows(
        buffer,
        ["name", "description", "items_json"],
        [{"name": "field_1A", "description": value, "items_json": inventory.json_text([value])}],
    )
    parsed = list(csv.DictReader(io.StringIO(buffer.getvalue())))
    assert parsed[0]["description"] == value
    assert json.loads(parsed[0]["items_json"]) == [value]


def test_existing_output_target_is_refused():
    with patch.object(Path, "exists", return_value=True):
        with pytest.raises(FileExistsError):
            inventory.validate_output_targets(Path("unused"))


def test_metadata_adapter_uses_metadata_and_schema_only():
    class FakeParquetFile:
        metadata = SimpleNamespace(num_rows=17, num_row_groups=3)
        schema_arrow = pa.schema([pa.field("case_id", pa.int64()), pa.field("nested", pa.list_(pa.int32()))])

        def read(self):
            raise AssertionError("row-data read API must not be called")

        def read_row_group(self, *args, **kwargs):
            raise AssertionError("row-group read API must not be called")

    result = inventory.read_parquet_metadata(
        Path("synthetic.parquet"), lambda unused_path: FakeParquetFile()
    )
    assert result == {
        "metadata_rows": 17,
        "row_group_count": 3,
        "fields": [
            {"position": 0, "name": "case_id", "arrow_type": "int64"},
            {"position": 1, "name": "nested", "arrow_type": "list<item: int32>"},
        ],
    }

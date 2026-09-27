import duckdb

from mvo.molecular_audit import key_stats, natural_key, local_path


def test_duplicate_and_null_key_are_independent():
    con = duckdb.connect()
    con.execute("create table t(lane varchar, gene varchar)")
    con.execute("insert into t values ('a','x'),('a','x'),('b','x'),('a',NULL)")
    result = key_stats(con, "t", ["lane", "gene"])
    assert result["duplicate_rows"] == 1
    assert result["null_or_blank_key_rows"] == 1
    assert result["status"] != "pass"


def test_keys_keep_caller_namespace():
    cols = ["cohort_run_id", "run_id", "proteome_id", "source_tool", "gene_id"]
    assert "source_tool" in natural_key("dim_gene", cols)
    assert natural_key("uninterpreted_table", ["proteome_id"]) == []


def test_unknown_root_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        local_path(tmp_path, "/unapproved/source/file")

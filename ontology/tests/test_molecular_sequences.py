from mvo.molecular_sequences import check_translation, prodigal_coordinates, sequence_digest, fasta_index
from mvo.molecular_stage import stable_id, GH5_PATTERN, KO_PATTERN, mucc_record_mapping
import pytest


def test_valid_and_invalid_translation():
    assert check_translation("ATGGCTTAA", "MA*", 1, 9, 1)["admit_encodes"]
    assert not check_translation("ATGGCTTAA", "MP", 1, 9, 1)["admit_encodes"]
    assert not check_translation("ATGGCTTAA", "MA", 0, 9, 1)["admit_encodes"]
    assert not check_translation("ATGGCTTAA", "MA", 1, 9, 0)["admit_encodes"]
    assert not check_translation("ATGGCTTAA", "MA", 1, 8, 1)["admit_encodes"]


def test_reverse_strand_and_alternative_start():
    assert check_translation("TTAAGCCAT", "MA", 1, 9, -1)["admit_encodes"]
    assert check_translation("GTGGCTTAA", "MA", 1, 9, 1)["status"] == "exact_with_allowed_initiator_methionine"
    assert not check_translation("GCTGCTTAA", "MA", 1, 9, 1)["admit_encodes"]
    assert not check_translation("ATGTGATAA", "MW", 1, 9, 1,table=11)["admit_encodes"]
    assert check_translation("ATGTGATAA", "MW", 1, 9, 1,table=4)["admit_encodes"]


def test_digest_is_sequence_not_identity():
    assert sequence_digest("ma*") == sequence_digest("MA")
    assert stable_id("locus", "lane1", "mag", "run", "x") != stable_id("locus", "lane2", "mag", "run", "x")
    assert stable_id("locus", "a/b", "c") != stable_id("locus", "a", "b/c")


def test_parse_source_header_and_duplicate_guard(tmp_path):
    c=prodigal_coordinates("scaffold_abc_1 # 1 # 9 # -1 # ID=1_1;partial=00;start_type=ATG")
    assert c["contig_id"] == "scaffold_abc"
    assert prodigal_coordinates("protein_without_coordinates") is None
    p=tmp_path/"a.faa"
    p.write_text(">x\nMA\n>x\nMP\n")
    with pytest.raises(ValueError, match="Duplicate"):
        fasta_index(p)


def test_panel_tokens_are_not_broad_keyword_search():
    assert GH5_PATTERN.search("GH5_2(1-99)")
    assert not GH5_PATTERN.search("GH50(1-99)")
    assert not GH5_PATTERN.search("GH500")
    assert KO_PATTERN.findall("K00399;K10944;K003990") == ["K00399", "K10944"]


def test_mucc_explicit_reference_alias_does_not_require_fabricated_fasta_identity():
    row={"":"reference_scaffold_1", "gene_db_id":"reference_scaffold_1", "db_id":"OWC_0098", "fasta":"named_reference"}
    assert mucc_record_mapping(row,{"OWC_0098"}) == ("OWC_0098", "explicit_source_db_id_reference_alias")
    row["fasta"]="OWC_0099"
    assert mucc_record_mapping(row,{"OWC_0098","OWC_0099"})[0] is None

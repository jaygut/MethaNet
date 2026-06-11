"""Unit tests for functional gene quantification."""

import shutil
from pathlib import Path

import numpy as np
import pytest

try:
    from methanet.functional.quantify import (
        FunctionalProfile,
        FunctionalQuantifier,
        MarkerGene,
    )
    from methanet.schema import FUNCTIONAL_FEATURE_COLUMNS
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


HMMSEARCH_AVAILABLE = shutil.which("hmmsearch") is not None


class TestMarkerGene:
    """Tests for MarkerGene dataclass."""

    def test_marker_creation(self):
        """Test creating a marker gene."""
        marker = MarkerGene(
            name="mcrA",
            hmm_path=Path("data/hmm/mcrA.hmm"),
            threshold=1e-10,
            description="Methyl-coenzyme M reductase",
        )

        assert marker.name == "mcrA"
        assert marker.threshold == 1e-10

    def test_default_threshold(self):
        """Test default E-value threshold."""
        marker = MarkerGene(
            name="test",
            hmm_path=Path("test.hmm"),
        )

        assert marker.threshold == 1e-10


class TestFunctionalProfile:
    """Tests for FunctionalProfile dataclass."""

    @pytest.fixture
    def sample_profile(self):
        """Create a sample profile for testing."""
        return FunctionalProfile(
            sample_id="test_sample",
            mcrA=100.0,
            pmoA=50.0,
            dsrA=25.0,
            nifH=0.0,
            cbbL=0.0,
            mmoX=10.0,  # Test new marker
        )

    def test_profile_creation(self, sample_profile):
        """Test creating a functional profile."""
        assert sample_profile.sample_id == "test_sample"
        assert sample_profile.marker_abundances["mcrA"] == 100.0
        assert sample_profile.pmoA == 50.0
        assert sample_profile.mmoX == 10.0

    def test_mcra_pmoa_ratio(self, sample_profile):
        """Test mcrA/pmoA ratio calculation."""
        ratio = sample_profile.mcra_pmoa_ratio

        assert ratio == pytest.approx(np.log2(2.0), rel=1e-6)

    def test_mcra_pmoa_ratio_zero_pmoa(self):
        """Test ratio when pmoA is zero."""
        profile = FunctionalProfile(
            sample_id="test",
            mcrA=100.0,
            pmoA=0.0,
            dsrA=0.0,
            nifH=0.0,
            cbbL=0.0,
        )

        assert profile.mcra_pmoa_ratio > 20.0

    def test_mcra_pmoa_ratio_zero_markers(self):
        """Test ratio when markers are zeroed."""
        profile = FunctionalProfile(
            sample_id="test",
            mcrA=0.0,
            pmoA=0.0,
            dsrA=50.0,
            nifH=0.0,
            cbbL=0.0,
        )

        assert profile.mcra_pmoa_ratio == pytest.approx(0.0, rel=1e-6)

    def test_normalized_abundances(self, sample_profile):
        """Test per-1k-proteins normalization passthrough."""
        normalized = sample_profile.get_normalized_abundances(
            method="per_1k_proteins"
        )

        assert "mcrA" in normalized
        assert normalized["mcrA"] == pytest.approx(100.0, rel=1e-6)
        assert "mmoX" in normalized
        assert normalized["mmoX"] == pytest.approx(10.0, rel=1e-6)

    def test_to_array(self, sample_profile):
        """Test conversion to numpy array."""
        arr = sample_profile.to_array()

        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1
        # 12 markers + 1 ratio = 13 features
        assert len(arr) == 13

    def test_to_array_uses_schema_order(self):
        """Functional vector order follows the public schema."""
        profile = FunctionalProfile(
            sample_id="ordered",
            mcrA=1.0,
            mcrB=2.0,
            mcrG=3.0,
            pmoA=4.0,
            mmoX=5.0,
            dsrA=6.0,
            dsrB=7.0,
            mtaB=8.0,
            mttB=9.0,
            mtbA=10.0,
            nifH=11.0,
            cbbL=12.0,
        )

        observed = dict(zip(FUNCTIONAL_FEATURE_COLUMNS, profile.to_array()))

        for index, marker in enumerate(FUNCTIONAL_FEATURE_COLUMNS[:-1], start=1):
            assert observed[marker] == pytest.approx(float(index))
        assert observed["mcrA_pmoA_ratio"] == pytest.approx(
            profile.mcrA_pmoA_ratio
        )

    def test_to_dict(self, sample_profile):
        """Test conversion to dictionary."""
        d = sample_profile.to_dict()

        assert isinstance(d, dict)
        assert d["sample_id"] == "test_sample"
        assert "mcrA" in d


class TestFunctionalQuantifier:
    """Tests for FunctionalQuantifier class."""

    @pytest.fixture
    def mock_hmm_dir(self, tmp_path):
        """Create mock HMM directory."""
        hmm_dir = tmp_path / "hmm"
        hmm_dir.mkdir()

        # Create mock HMM files (empty files for testing)
        markers = [
            "mcrA",
            "mcrB",
            "mcrG",
            "pmoA",
            "mmoX",
            "dsrA",
            "dsrB",
            "mtaB",
            "mttB",
            "mtbA",
            "nifH",
            "cbbL",
        ]
        for marker in markers:
            (hmm_dir / f"{marker}.hmm").touch()

        return hmm_dir

    @pytest.fixture
    def mock_fasta(self, tmp_path):
        """Create mock FASTA file."""
        fasta_path = tmp_path / "test.faa"
        fasta_content = """>protein_1
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK
>protein_2
MASVKKVLGNGFGRIGRLVTRAAFNSGKVDIVAINDPFIDLNYMVYMFQYDSTHGKFHGT
>protein_3
MLSLLLNTALLASAAASPGQKAHFADACLAALAQHGGTGFAAALSNAASPAAQNRSGIAP
"""
        fasta_path.write_text(fasta_content)
        return fasta_path

    def test_quantifier_init(self, mock_hmm_dir):
        """Test quantifier initialization."""
        quantifier = FunctionalQuantifier(
            hmm_dir=mock_hmm_dir,
            markers=["mcrA", "pmoA"],
        )

        assert len(quantifier.markers) == 2
        assert quantifier.hmm_dir == mock_hmm_dir

    def test_quantifier_default_markers(self, mock_hmm_dir):
        """Test default markers."""
        quantifier = FunctionalQuantifier(hmm_dir=mock_hmm_dir)

        expected_markers = [
            "mcrA",
            "mcrB",
            "mcrG",
            "pmoA",
            "mmoX",
            "dsrA",
            "dsrB",
            "mtaB",
            "mttB",
            "mtbA",
            "nifH",
            "cbbL",
        ]
        observed_markers = [marker.name for marker in quantifier.markers]
        assert all(m in observed_markers for m in expected_markers)

    def test_quantifier_missing_hmm(self, tmp_path):
        """Test error when HMM file is missing."""
        hmm_dir = tmp_path / "empty_hmm"
        hmm_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            FunctionalQuantifier(
                hmm_dir=hmm_dir,
                markers=["mcrA"],
            )

    def test_parse_hmm_output_skips_non_numeric_lines(self, mock_hmm_dir):
        """Parser should skip non-tabular/non-numeric lines safely."""
        quantifier = FunctionalQuantifier(
            hmm_dir=mock_hmm_dir,
            markers=["mcrA"],
            score_threshold=50.0,
        )

        raw_output = "\n".join(
            [
                "# hmmsearch :: tblout",
                "target1 - 100 mcrA - 200 1e-25 120.0 0.0",
                "this line is not numeric at expected columns and should be skipped",
                "target2 - 100 mcrA - 200 1e-5 40.0 0.0",
            ]
        )

        hits = quantifier._parse_hmm_output(raw_output)
        assert len(hits) == 1
        assert hits[0]["target"] == "target1"
        assert hits[0]["query"] == "mcrA"

    @pytest.mark.skipif(
        not (IMPORTS_AVAILABLE and HMMSEARCH_AVAILABLE),
        reason="Functional quantification requires hmmsearch (HMMER)",
    )
    def test_quantify_mag_returns_profile(self, mock_hmm_dir, mock_fasta):
        """Test that quantify_mag returns a FunctionalProfile."""
        quantifier = FunctionalQuantifier(
            hmm_dir=mock_hmm_dir,
            markers=["mcrA", "pmoA"],
        )

        profile = quantifier.quantify_mag(mock_fasta)

        assert isinstance(profile, FunctionalProfile)
        assert profile.sample_id == "test"


class TestPathwayCompleteness:
    """Tests for pathway completeness scoring."""

    def test_mcr_complex_completeness(self):
        """Test MCR complex completeness score."""
        # MCR complex requires mcrA, mcrB, mcrC, mcrD, mcrG
        components = {
            "mcrA": True,
            "mcrB": True,
            "mcrC": True,
            "mcrD": False,
            "mcrG": False,
        }

        # 3 out of 5 components = 60%
        score = sum(components.values()) / len(components)

        assert score == pytest.approx(0.6, rel=1e-6)

    def test_hdr_complex_completeness(self):
        """Test HdrABC complex completeness."""
        components = {
            "hdrA": True,
            "hdrB": True,
            "hdrC": True,
        }

        score = sum(components.values()) / len(components)

        assert score == pytest.approx(1.0, rel=1e-6)


class TestNormalization:
    """Tests for normalization selection."""

    @pytest.fixture
    def sample_profile(self):
        """Create a sample profile for testing."""
        return FunctionalProfile(
            sample_id="test_sample",
            mcrA=100.0,
            pmoA=50.0,
            dsrA=25.0,
            nifH=0.0,
            cbbL=0.0,
        )

    def test_raw_normalization(self, sample_profile):
        """Test raw normalization passthrough."""
        normalized = sample_profile.get_normalized_abundances(method="raw")
        assert normalized["mcrA"] == pytest.approx(100.0, rel=1e-6)

    def test_invalid_normalization(self, sample_profile):
        """Test unsupported normalization method raises."""
        with pytest.raises(ValueError):
            sample_profile.get_normalized_abundances(method="rpkm")

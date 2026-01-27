"""Unit tests for functional gene quantification."""

from pathlib import Path
import shutil

import numpy as np
import pytest

try:
    from methanet.functional.quantify import (
        FunctionalProfile,
        FunctionalQuantifier,
        MarkerGene,
    )
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
        )

    def test_profile_creation(self, sample_profile):
        """Test creating a functional profile."""
        assert sample_profile.sample_id == "test_sample"
        assert sample_profile.marker_abundances["mcrA"] == 100.0
        assert sample_profile.pmoA == 50.0

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

    def test_to_array(self, sample_profile):
        """Test conversion to numpy array."""
        arr = sample_profile.to_array()

        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1

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
        for marker in ["mcrA", "pmoA", "dsrA", "nifH", "cbbL"]:
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

        expected_markers = ["mcrA", "pmoA", "dsrA", "nifH", "cbbL"]
        assert all(m in [marker.name for marker in quantifier.markers]
                   for m in expected_markers)

    def test_quantifier_missing_hmm(self, tmp_path):
        """Test error when HMM file is missing."""
        hmm_dir = tmp_path / "empty_hmm"
        hmm_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            FunctionalQuantifier(
                hmm_dir=hmm_dir,
                markers=["mcrA"],
            )

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

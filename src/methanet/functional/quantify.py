"""Functional gene quantification using HMM search.

This module implements HMM-based quantification of methane-related
functional gene markers in metagenome-assembled genomes (MAGs).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import subprocess
import numpy as np


@dataclass
class MarkerGene:
    """Marker gene definition for HMM-based searches."""

    name: str
    hmm_path: Path
    threshold: float = 1e-10
    description: Optional[str] = None


@dataclass
class FunctionalProfile:
    """Functional gene abundance profile for a MAG.

    Attributes:
        sample_id: Unique identifier for the sample/MAG.
        mcrA: Methyl-coenzyme M reductase alpha subunit abundance.
        mcrB: Methyl-coenzyme M reductase beta subunit abundance.
        mcrG: Methyl-coenzyme M reductase gamma subunit abundance.
        pmoA: Particulate methane monooxygenase subunit A abundance.
        mmoX: Soluble methane monooxygenase component A alpha.
        dsrA: Dissimilatory sulfite reductase alpha subunit abundance.
        dsrB: Dissimilatory sulfite reductase beta subunit abundance.
        nifH: Nitrogenase iron protein abundance.
        cbbL: RuBisCO large subunit abundance.
        mtaB: Methanol-cobalamin methyltransferase abundance.
        mttB: Trimethylamine methyltransferase abundance.
        mtbA: Methylcobalamin:CoM methyltransferase abundance.
    """

    sample_id: str
    mcrA: float
    pmoA: float
    dsrA: float
    nifH: float
    cbbL: float
    # New markers
    mcrB: float = 0.0
    mcrG: float = 0.0
    mmoX: float = 0.0
    dsrB: float = 0.0
    mtaB: float = 0.0
    mttB: float = 0.0
    mtbA: float = 0.0

    @property
    def marker_abundances(self) -> Dict[str, float]:
        """Return marker abundances as a dictionary."""
        return {
            "mcrA": self.mcrA,
            "mcrB": self.mcrB,
            "mcrG": self.mcrG,
            "pmoA": self.pmoA,
            "mmoX": self.mmoX,
            "dsrA": self.dsrA,
            "dsrB": self.dsrB,
            "nifH": self.nifH,
            "cbbL": self.cbbL,
            "mtaB": self.mtaB,
            "mttB": self.mttB,
            "mtbA": self.mtbA,
        }

    @property
    def mcrA_pmoA_ratio(self) -> float:
        """Log2-transformed mcrA/pmoA ratio with pseudocount.

        Positive values indicate methanogenic potential,
        negative values indicate methanotrophic dominance.
        """
        pseudocount = 1e-6
        # Use primary markers for the main ratio
        return np.log2((self.mcrA + pseudocount) / (self.pmoA + pseudocount))

    @property
    def mcra_pmoa_ratio(self) -> float:
        """Alias for mcrA_pmoA_ratio (backward compatibility)."""
        return self.mcrA_pmoA_ratio

    @property
    def methanogenic_potential(self) -> str:
        """Categorical methanogenic potential classification.

        Returns:
            One of: 'high', 'moderate', 'low', 'methanotrophic'
        """
        ratio = self.mcrA_pmoA_ratio
        if ratio > 2.0:
            return "high"
        elif ratio > 0.0:
            return "moderate"
        elif ratio > -2.0:
            return "low"
        else:
            return "methanotrophic"

    def to_vector(self) -> np.ndarray:
        """Convert profile to feature vector for ML models.

        Returns:
            Array of [mcrA, pmoA, dsrA, nifH, cbbL, ratio, ...new_markers]
        """
        return np.array([
            self.mcrA,
            self.pmoA,
            self.dsrA,
            self.nifH,
            self.cbbL,
            self.mcrA_pmoA_ratio,
            # New markers added to end of vector
            self.mcrB,
            self.mcrG,
            self.mmoX,
            self.dsrB,
            self.mtaB,
            self.mttB,
            self.mtbA,
        ])

    def to_array(self) -> np.ndarray:
        """Alias for to_vector."""
        return self.to_vector()

    def get_normalized_abundances(self, method: str = "per_1k_proteins") -> Dict[str, float]:
        """Return normalized abundances using a supported method."""
        method = method.lower()
        if method in {"per_1k_proteins", "raw"}:
            return self.marker_abundances
        raise ValueError(f"Unsupported normalization method: {method}")

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "sample_id": self.sample_id,
            "mcrA": self.mcrA,
            "pmoA": self.pmoA,
            "dsrA": self.dsrA,
            "nifH": self.nifH,
            "cbbL": self.cbbL,
            "mcrB": self.mcrB,
            "mcrG": self.mcrG,
            "mmoX": self.mmoX,
            "dsrB": self.dsrB,
            "mtaB": self.mtaB,
            "mttB": self.mttB,
            "mtbA": self.mtbA,
            "mcrA_pmoA_ratio": self.mcrA_pmoA_ratio,
            "methanogenic_potential": self.methanogenic_potential,
        }


# HMM profile to gene name mapping
GENE_MAPPING = {
    # Core Methanogenesis
    "PF02249": "mcrA",
    "TIGR03256": "mcrA",
    "TIGR03258": "mcrB",
    "TIGR03259": "mcrG",
    # Oxidizers
    "PF02461": "pmoA",
    "TIGR03080": "pmoA",
    "TIGR01691": "mmoX",
    # Competitors
    "PF04358": "dsrA",
    "TIGR02064": "dsrA",
    "TIGR02066": "dsrB",
    # Methylotrophic / Saline
    "TIGR02626": "mtaB",
    "TIGR02512": "mttB",
    "TIGR02506": "mtbA",
    # Controls
    "PF00142": "nifH",
    "TIGR01287": "nifH",
    "PF00016": "cbbL",
    "TIGR01168": "cbbL",
}

DEFAULT_MARKERS = (
    "mcrA", "mcrB", "mcrG",
    "pmoA", "mmoX",
    "dsrA", "dsrB",
    "nifH", "cbbL",
    "mtaB", "mttB", "mtbA"
)


class FunctionalQuantifier:
    """Quantify functional gene markers using HMM search.

    This class provides methods to run hmmsearch against a database
    of functional gene HMM profiles and compute normalized abundances.

    Attributes:
        hmm_db_path: Path to HMM database file.
        hmm_dir: Directory containing per-marker HMMs.
        evalue_threshold: Maximum e-value for valid hits.
        score_threshold: Minimum bit score for valid hits.
        threads: Number of CPU threads for hmmsearch.
    """

    def __init__(
        self,
        hmm_db_path: Optional[Path] = None,
        hmm_dir: Optional[Path] = None,
        markers: Optional[Sequence[Union[str, MarkerGene]]] = None,
        evalue_threshold: float = 1e-10,
        score_threshold: float = 50.0,
        threads: int = 8,
    ):
        """Initialize the quantifier.

        Args:
            hmm_db_path: Path to concatenated HMM profile database.
            hmm_dir: Directory containing per-marker HMMs.
            markers: Marker names or MarkerGene definitions (defaults to core markers).
            evalue_threshold: Maximum e-value for reporting hits.
            score_threshold: Minimum bit score for counting hits.
            threads: Number of CPU threads for parallel search.
        """
        if hmm_db_path and hmm_dir:
            raise ValueError("Provide either hmm_db_path or hmm_dir, not both.")
        if not hmm_db_path and not hmm_dir:
            raise ValueError("Provide hmm_db_path or hmm_dir.")

        self.hmm_db_path = Path(hmm_db_path) if hmm_db_path else None
        self.hmm_dir = Path(hmm_dir) if hmm_dir else None
        self.evalue_threshold = evalue_threshold
        self.score_threshold = score_threshold
        self.threads = threads
        self._profile_lengths: Dict[str, int] = {}
        self.markers: List[MarkerGene] = []

        if self.hmm_dir:
            self.markers = self._resolve_markers(markers)

    def quantify(
        self,
        protein_fasta: Path,
        sample_id: str,
    ) -> FunctionalProfile:
        """Run HMM search and compute normalized abundances.

        Args:
            protein_fasta: Path to predicted protein sequences (FASTA).
            sample_id: Identifier for this sample.

        Returns:
            FunctionalProfile with normalized gene abundances.
        """
        # Run hmmsearch
        if self.hmm_db_path:
            hits = self._run_hmmsearch_db(protein_fasta)
            abundances = self._aggregate_hits(hits)
        else:
            abundances = self._run_marker_searches(protein_fasta)

        # Normalize by genome size (proteins per Mb proxy)
        total_proteins = self._count_proteins(protein_fasta)
        normalization_factor = total_proteins / 1000 if total_proteins > 0 else 1

        normalized = {
            gene: count / normalization_factor for gene, count in abundances.items()
        }

        return FunctionalProfile(
            sample_id=sample_id,
            mcrA=normalized.get("mcrA", 0.0),
            mcrB=normalized.get("mcrB", 0.0),
            mcrG=normalized.get("mcrG", 0.0),
            pmoA=normalized.get("pmoA", 0.0),
            mmoX=normalized.get("mmoX", 0.0),
            dsrA=normalized.get("dsrA", 0.0),
            dsrB=normalized.get("dsrB", 0.0),
            nifH=normalized.get("nifH", 0.0),
            cbbL=normalized.get("cbbL", 0.0),
            mtaB=normalized.get("mtaB", 0.0),
            mttB=normalized.get("mttB", 0.0),
            mtbA=normalized.get("mtbA", 0.0),
        )

    def quantify_mag(self, protein_fasta: Path, sample_id: Optional[str] = None) -> FunctionalProfile:
        """Convenience wrapper for MAG quantification."""
        inferred_id = sample_id or Path(protein_fasta).stem
        return self.quantify(protein_fasta, inferred_id)

    def _resolve_markers(
        self, markers: Optional[Sequence[Union[str, MarkerGene]]]
    ) -> List[MarkerGene]:
        if markers is None:
            markers = DEFAULT_MARKERS

        resolved: List[MarkerGene] = []
        for marker in markers:
            if isinstance(marker, MarkerGene):
                resolved.append(marker)
                continue
            hmm_path = self.hmm_dir / f"{marker}.hmm"
            resolved.append(
                MarkerGene(
                    name=str(marker),
                    hmm_path=hmm_path,
                    threshold=self.evalue_threshold,
                )
            )

        for marker in resolved:
            if not marker.hmm_path.exists():
                raise FileNotFoundError(f"HMM file not found: {marker.hmm_path}")

        return resolved

    def _run_marker_searches(self, protein_fasta: Path) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for marker in self.markers:
            counts[marker.name] = self._run_hmmsearch_marker(
                protein_fasta, marker
            )
        return counts

    def _run_hmmsearch_db(self, protein_fasta: Path) -> List[Dict]:
        """Execute hmmsearch and parse results.

        Args:
            protein_fasta: Path to query protein sequences.

        Returns:
            List of hit dictionaries with target, query, evalue, score.
        """
        if not self.hmm_db_path:
            raise RuntimeError("hmm_db_path is required for database searches.")
        cmd = [
            "hmmsearch",
            "--tblout",
            "/dev/stdout",
            "-E",
            str(self.evalue_threshold),
            "--cpu",
            str(self.threads),
            str(self.hmm_db_path),
            str(protein_fasta),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            return self._parse_hmm_output(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"hmmsearch failed: {e.stderr}") from e
        except FileNotFoundError:
            raise RuntimeError(
                "hmmsearch not found. Install HMMER: conda install -c bioconda hmmer"
            )

    def _run_hmmsearch_marker(self, protein_fasta: Path, marker: MarkerGene) -> int:
        cmd = [
            "hmmsearch",
            "--tblout",
            "/dev/stdout",
            "-E",
            str(marker.threshold),
            "--cpu",
            str(self.threads),
            str(marker.hmm_path),
            str(protein_fasta),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            hits = self._parse_hmm_output(result.stdout, self.score_threshold)
            return self._count_unique_targets(hits)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"hmmsearch failed: {e.stderr}") from e
        except FileNotFoundError:
            raise RuntimeError(
                "hmmsearch not found. Install HMMER: conda install -c bioconda hmmer"
            )

    def _parse_hmm_output(self, output: str, score_threshold: Optional[float] = None) -> List[Dict]:
        """Parse hmmsearch tabular output.

        Args:
            output: Raw hmmsearch tblout format string.

        Returns:
            List of parsed hit dictionaries.
        """
        threshold = self.score_threshold if score_threshold is None else score_threshold
        hits = []
        for line in output.strip().split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) >= 8:
                hit = {
                    "target": fields[0],
                    "query": fields[3],
                    "evalue": float(fields[6]),
                    "score": float(fields[7]),
                }
                if hit["score"] >= threshold:
                    hits.append(hit)
        return hits

    def _count_unique_targets(self, hits: List[Dict]) -> int:
        return len({hit["target"] for hit in hits})

    def _aggregate_hits(self, hits: List[Dict]) -> Dict[str, int]:
        """Map HMM profiles to gene names and count unique hits.

        Args:
            hits: List of parsed HMM hits.

        Returns:
            Dictionary of gene name to hit count.
        """
        counts: Dict[str, int] = {}
        seen_targets: set = set()

        # Sort by score descending to keep best hits
        for hit in sorted(hits, key=lambda x: x["score"], reverse=True):
            if hit["target"] in seen_targets:
                continue
            seen_targets.add(hit["target"])

            # Extract profile ID from query name
            profile_id = hit["query"].split(".")[0]
            gene = GENE_MAPPING.get(profile_id)
            if gene:
                counts[gene] = counts.get(gene, 0) + 1

        return counts

    def _count_proteins(self, fasta: Path) -> int:
        """Count protein sequences in FASTA file.

        Args:
            fasta: Path to FASTA file.

        Returns:
            Number of sequences (header lines).
        """
        count = 0
        with open(fasta) as f:
            for line in f:
                if line.startswith(">"):
                    count += 1
        return count

    def quantify_batch(
        self,
        samples: List[tuple],
    ) -> List[FunctionalProfile]:
        """Quantify multiple samples.

        Args:
            samples: List of (protein_fasta_path, sample_id) tuples.

        Returns:
            List of FunctionalProfile objects.
        """
        return [
            self.quantify(Path(fasta), sample_id) for fasta, sample_id in samples
        ]

"""Functional gene quantification using HMM search.

This module implements HMM-based quantification of methane-related
functional gene markers in metagenome-assembled genomes (MAGs).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import numpy as np


@dataclass
class FunctionalProfile:
    """Functional gene abundance profile for a MAG.

    Attributes:
        sample_id: Unique identifier for the sample/MAG.
        mcrA: Methyl-coenzyme M reductase alpha subunit abundance.
        pmoA: Particulate methane monooxygenase subunit A abundance.
        dsrA: Dissimilatory sulfite reductase alpha subunit abundance.
        nifH: Nitrogenase iron protein abundance.
        cbbL: RuBisCO large subunit abundance.
    """

    sample_id: str
    mcrA: float
    pmoA: float
    dsrA: float
    nifH: float
    cbbL: float

    @property
    def mcrA_pmoA_ratio(self) -> float:
        """Log2-transformed mcrA/pmoA ratio with pseudocount.

        Positive values indicate methanogenic potential,
        negative values indicate methanotrophic dominance.
        """
        pseudocount = 1e-6
        return np.log2((self.mcrA + pseudocount) / (self.pmoA + pseudocount))

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
            Array of [mcrA, pmoA, dsrA, nifH, cbbL, ratio]
        """
        return np.array([
            self.mcrA,
            self.pmoA,
            self.dsrA,
            self.nifH,
            self.cbbL,
            self.mcrA_pmoA_ratio,
        ])

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "sample_id": self.sample_id,
            "mcrA": self.mcrA,
            "pmoA": self.pmoA,
            "dsrA": self.dsrA,
            "nifH": self.nifH,
            "cbbL": self.cbbL,
            "mcrA_pmoA_ratio": self.mcrA_pmoA_ratio,
            "methanogenic_potential": self.methanogenic_potential,
        }


# HMM profile to gene name mapping
GENE_MAPPING = {
    "PF02249": "mcrA",
    "TIGR03256": "mcrA",
    "PF02461": "pmoA",
    "TIGR03080": "pmoA",
    "PF04358": "dsrA",
    "TIGR02064": "dsrA",
    "PF00142": "nifH",
    "TIGR01287": "nifH",
    "PF00016": "cbbL",
    "TIGR01168": "cbbL",
}


class FunctionalQuantifier:
    """Quantify functional gene markers using HMM search.

    This class provides methods to run hmmsearch against a database
    of functional gene HMM profiles and compute normalized abundances.

    Attributes:
        hmm_db_path: Path to HMM database file.
        evalue_threshold: Maximum e-value for valid hits.
        score_threshold: Minimum bit score for valid hits.
        threads: Number of CPU threads for hmmsearch.
    """

    def __init__(
        self,
        hmm_db_path: Path,
        evalue_threshold: float = 1e-10,
        score_threshold: float = 50.0,
        threads: int = 8,
    ):
        """Initialize the quantifier.

        Args:
            hmm_db_path: Path to concatenated HMM profile database.
            evalue_threshold: Maximum e-value for reporting hits.
            score_threshold: Minimum bit score for counting hits.
            threads: Number of CPU threads for parallel search.
        """
        self.hmm_db_path = Path(hmm_db_path)
        self.evalue_threshold = evalue_threshold
        self.score_threshold = score_threshold
        self.threads = threads
        self._profile_lengths: Dict[str, int] = {}

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
        hits = self._run_hmmsearch(protein_fasta)

        # Aggregate hits by gene
        abundances = self._aggregate_hits(hits)

        # Normalize by genome size (proteins per Mb proxy)
        total_proteins = self._count_proteins(protein_fasta)
        normalization_factor = total_proteins / 1000 if total_proteins > 0 else 1

        normalized = {
            gene: count / normalization_factor for gene, count in abundances.items()
        }

        return FunctionalProfile(
            sample_id=sample_id,
            mcrA=normalized.get("mcrA", 0.0),
            pmoA=normalized.get("pmoA", 0.0),
            dsrA=normalized.get("dsrA", 0.0),
            nifH=normalized.get("nifH", 0.0),
            cbbL=normalized.get("cbbL", 0.0),
        )

    def _run_hmmsearch(self, protein_fasta: Path) -> List[Dict]:
        """Execute hmmsearch and parse results.

        Args:
            protein_fasta: Path to query protein sequences.

        Returns:
            List of hit dictionaries with target, query, evalue, score.
        """
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

    def _parse_hmm_output(self, output: str) -> List[Dict]:
        """Parse hmmsearch tabular output.

        Args:
            output: Raw hmmsearch tblout format string.

        Returns:
            List of parsed hit dictionaries.
        """
        hits = []
        for line in output.strip().split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) >= 9:
                hit = {
                    "target": fields[0],
                    "query": fields[2],
                    "evalue": float(fields[4]),
                    "score": float(fields[5]),
                }
                if hit["score"] >= self.score_threshold:
                    hits.append(hit)
        return hits

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

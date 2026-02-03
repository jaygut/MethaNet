"""GenomeOcean genomic foundation model embeddings.

This module provides genome-level embeddings using the GenomeOcean
foundation model (3072-dim). Note: GenomeOcean requires separate
model weights that must be obtained from the original authors.

For systems without GenomeOcean access, DNABERT-2 can be used as
an alternative via the workflow scripts.
"""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class GenomeOceanConfig:
    """Configuration for GenomeOcean embedding generation.

    Attributes:
        model_path: Path to GenomeOcean model checkpoint.
        batch_size: Batch size for inference.
        kmer_size: k-mer size for tokenization (default: 6).
        device: Compute device ('cuda', 'cpu', or 'auto').
        fp16: Use half-precision inference.
    """

    model_path: Optional[Path] = None
    batch_size: int = 4
    kmer_size: int = 6
    device: str = "auto"
    fp16: bool = True
    allow_stub: bool = False

    def __post_init__(self):
        if self.device == "auto":
            self.device = (
                "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
            )


class GenomeOceanEmbedder:
    """Generate genome embeddings using GenomeOcean foundation model.

    GenomeOcean is a 4B parameter model trained on diverse genomic data.
    It produces 3072-dimensional embeddings suitable for downstream tasks.

    Note: This requires the GenomeOcean model weights. If not available,
    the workflow uses DNABERT-2 as an alternative.
    """

    EMBEDDING_DIM = 3072

    def __init__(self, config: Optional[GenomeOceanConfig] = None):
        """Initialize GenomeOcean embedder.

        Args:
            config: Embedding configuration.

        Raises:
            ImportError: If PyTorch is not available.
            FileNotFoundError: If model path doesn't exist.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch required. Install: pip install torch"
            )

        self.config = config or GenomeOceanConfig()
        self.device = torch.device(self.config.device)
        self._vocab = self._build_kmer_vocab()

        # Model placeholder - actual model requires GenomeOcean weights
        self.model = None
        if self.config.model_path and Path(self.config.model_path).exists():
            self.model = self._load_model(self.config.model_path)

    def _load_model(self, model_path: Path) -> nn.Module:
        """Load GenomeOcean model from checkpoint.

        Args:
            model_path: Path to model checkpoint.

        Returns:
            Loaded PyTorch model.
        """
        checkpoint = torch.load(model_path, map_location="cpu")
        # GenomeOcean uses a custom architecture
        # This is a placeholder - actual implementation depends on
        # GenomeOcean's released model format
        model = self._create_model_placeholder()
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)

        if self.config.fp16 and self.device.type == "cuda":
            model = model.half()

        model.eval()
        return model

    def _create_model_placeholder(self) -> nn.Module:
        """Create placeholder model structure.

        Returns:
            Simple transformer-based model for testing.
        """
        # Placeholder architecture for when model weights unavailable
        return nn.Sequential(
            nn.Embedding(len(self._vocab), 256),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=256, nhead=8),
                num_layers=2,
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, self.EMBEDDING_DIM),
        )

    def _build_kmer_vocab(self) -> Dict[str, int]:
        """Build k-mer vocabulary mapping.

        Returns:
            Dictionary mapping k-mers to token indices.
        """
        bases = "ACGT"
        k = self.config.kmer_size
        kmers = ["".join(p) for p in product(bases, repeat=k)]

        vocab = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3}
        for i, kmer in enumerate(kmers):
            vocab[kmer] = i + 4

        return vocab

    def _tokenize_genome(self, sequence: str) -> "torch.Tensor":
        """Convert genome sequence to k-mer tokens.

        Args:
            sequence: DNA sequence string.

        Returns:
            Tensor of token indices.
        """
        k = self.config.kmer_size
        sequence = sequence.upper()

        # Extract k-mers
        kmers = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]

        # Convert to vocabulary indices
        tokens = [self._vocab.get(kmer, self._vocab["<UNK>"]) for kmer in kmers]

        return torch.tensor(tokens, dtype=torch.long)

    @torch.no_grad()
    def embed_genome(self, genome_fasta: Path) -> np.ndarray:
        """Generate embedding for a genome.

        Args:
            genome_fasta: Path to genome FASTA file.

        Returns:
            3072-dimensional embedding array.

        Raises:
            RuntimeError: If model not loaded.
        """
        if self.model is None:
            if not self.config.allow_stub:
                raise RuntimeError(
                    "GenomeOcean model not loaded. Provide model_path or "
                    "use DNABERT-2 as the genome embedding backend."
                )
            import warnings

            warnings.warn(
                "GenomeOcean model not loaded. Returning NaN embedding. "
                "Use DNABERT-2 via workflow scripts as alternative."
            )
            return np.full(self.EMBEDDING_DIM, np.nan)

        try:
            from Bio import SeqIO
        except ImportError:
            raise ImportError("BioPython required. Install: pip install biopython")

        # Concatenate all contigs
        full_sequence = ""
        for record in SeqIO.parse(genome_fasta, "fasta"):
            full_sequence += str(record.seq)

        if not full_sequence:
            return np.full(self.EMBEDDING_DIM, np.nan)

        # Tokenize
        tokens = self._tokenize_genome(full_sequence)
        tokens = tokens.unsqueeze(0).to(self.device)

        # Generate embedding
        embedding = self.model(tokens)

        return embedding.cpu().float().numpy().squeeze()

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.EMBEDDING_DIM

    @classmethod
    def is_available(cls) -> bool:
        """Check if GenomeOcean can be used.

        Returns:
            True if PyTorch is available.
        """
        return TORCH_AVAILABLE

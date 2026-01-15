"""ESM-2 protein language model embeddings.

This module provides protein-level and genome-level embeddings using
the ESM-2 foundation model (facebook/esm2_t33_650M_UR50D).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class EmbeddingConfig:
    """Configuration for ESM-2 embedding generation.

    Attributes:
        model_name: HuggingFace model identifier.
        batch_size: Number of sequences per batch.
        max_length: Maximum sequence length (truncated if longer).
        pooling_layers: Layer indices to use for pooling.
        pooling_strategy: How to aggregate token embeddings ('mean', 'cls', 'last').
        device: Compute device ('cuda', 'cpu', or 'auto').
        fp16: Use half-precision inference on GPU.
        cache_dir: Directory for model cache.
    """

    model_name: str = "facebook/esm2_t33_650M_UR50D"
    batch_size: int = 8
    max_length: int = 1024
    pooling_layers: Tuple[int, ...] = field(
        default_factory=lambda: tuple(range(20, 34))  # layers 20-33
    )
    pooling_strategy: str = "mean"
    device: str = "auto"
    fp16: bool = True
    cache_dir: Optional[Path] = None

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"


class ProteinDataset(Dataset):
    """PyTorch Dataset for protein sequences."""

    def __init__(
        self,
        sequences: List[str],
        ids: List[str],
        tokenizer,
        max_length: int,
    ):
        self.sequences = sequences
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Truncate if necessary (leave room for special tokens)
        if len(seq) > self.max_length - 2:
            seq = seq[: self.max_length - 2]

        encoding = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "id": self.ids[idx],
        }


class ESM2Embedder:
    """Generate protein embeddings using ESM-2.

    This class handles loading the ESM-2 model and generating
    embeddings for protein sequences with configurable pooling.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize ESM-2 embedder.

        Args:
            config: Embedding configuration. Uses defaults if None.
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and transformers required. "
                "Install: pip install torch transformers"
            )

        self.config = config or EmbeddingConfig()
        self.device = torch.device(self.config.device)

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            output_hidden_states=True,
        ).to(self.device)

        if self.config.fp16 and self.device.type == "cuda":
            self.model = self.model.half()

        self.model.eval()

    @torch.no_grad()
    def embed_proteins(
        self,
        sequences: List[str],
        ids: List[str],
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings for protein sequences.

        Args:
            sequences: List of protein sequences (amino acid strings).
            ids: List of sequence identifiers.

        Returns:
            Dictionary mapping sequence ID to embedding array.
        """
        dataset = ProteinDataset(
            sequences, ids, self.tokenizer, self.config.max_length
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=self.device.type == "cuda",
        )

        embeddings = {}

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            batch_ids = batch["id"]

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Extract hidden states from specified layers
            hidden_states = outputs.hidden_states
            selected_layers = [
                hidden_states[i] for i in self.config.pooling_layers
                if i < len(hidden_states)
            ]

            # Stack and average across layers
            stacked = torch.stack(selected_layers, dim=0)
            layer_mean = stacked.mean(dim=0)  # [batch, seq_len, hidden_dim]

            # Apply pooling strategy
            pooled = self._apply_pooling(layer_mean, attention_mask)

            # Store embeddings
            pooled_np = pooled.cpu().float().numpy()
            for i, seq_id in enumerate(batch_ids):
                embeddings[seq_id] = pooled_np[i]

        return embeddings

    def _apply_pooling(
        self,
        hidden_states: "torch.Tensor",
        attention_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Apply pooling strategy to hidden states.

        Args:
            hidden_states: Shape [batch, seq_len, hidden_dim]
            attention_mask: Shape [batch, seq_len]

        Returns:
            Pooled embeddings of shape [batch, hidden_dim]
        """
        if self.config.pooling_strategy == "mean":
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask
        elif self.config.pooling_strategy == "cls":
            return hidden_states[:, 0, :]
        elif self.config.pooling_strategy == "last":
            # Get last non-padded token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            return hidden_states[
                torch.arange(batch_size, device=self.device),
                seq_lengths.long(),
            ]
        else:
            raise ValueError(
                f"Unknown pooling strategy: {self.config.pooling_strategy}"
            )

    def embed_genome(
        self,
        protein_embeddings: Dict[str, np.ndarray],
        aggregation: str = "mean",
    ) -> np.ndarray:
        """Aggregate protein embeddings to genome-level.

        Args:
            protein_embeddings: Dictionary of protein ID to embedding.
            aggregation: Aggregation method ('mean', 'max', 'concat').

        Returns:
            Genome-level embedding array.
        """
        if not protein_embeddings:
            return np.zeros(self.model.config.hidden_size)

        emb_matrix = np.stack(list(protein_embeddings.values()))

        if aggregation == "mean":
            return emb_matrix.mean(axis=0)
        elif aggregation == "max":
            return emb_matrix.max(axis=0)
        elif aggregation == "concat":
            return np.concatenate([emb_matrix.mean(axis=0), emb_matrix.max(axis=0)])
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.model.config.hidden_size


def embed_mag(
    protein_fasta: Path,
    config: Optional[EmbeddingConfig] = None,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """End-to-end embedding generation for a MAG.

    Args:
        protein_fasta: Path to predicted proteins (FASTA format).
        config: Optional embedding configuration.
        output_path: Optional path to save embedding (.npy).

    Returns:
        Genome-level embedding array (1280-dim for ESM-2).
    """
    try:
        from Bio import SeqIO
    except ImportError:
        raise ImportError("BioPython required. Install: pip install biopython")

    # Parse proteins
    sequences = []
    ids = []
    for record in SeqIO.parse(protein_fasta, "fasta"):
        sequences.append(str(record.seq))
        ids.append(record.id)

    if not sequences:
        raise ValueError(f"No sequences found in {protein_fasta}")

    # Generate embeddings
    embedder = ESM2Embedder(config)
    protein_embs = embedder.embed_proteins(sequences, ids)
    genome_emb = embedder.embed_genome(protein_embs)

    # Save if output path provided
    if output_path:
        np.save(output_path, genome_emb)

    return genome_emb

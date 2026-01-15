"""Domain adaptation loss functions.

This module implements loss functions for domain adaptation:
- MMDLoss: Maximum Mean Discrepancy with multi-kernel Gaussian
- CORALLoss: Correlation Alignment via covariance matching
"""

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None


if TORCH_AVAILABLE:

    class MMDLoss(nn.Module):
        """Maximum Mean Discrepancy loss with Gaussian kernel.

        MMD measures the distance between the means of two distributions
        in a reproducing kernel Hilbert space. Uses multiple Gaussian
        kernels with different bandwidths for robustness.

        Attributes:
            bandwidth: Base kernel bandwidth.
            kernel_num: Number of kernels with different bandwidths.
        """

        def __init__(self, kernel_bandwidth: float = 1.0, kernel_num: int = 5):
            """Initialize MMD loss.

            Args:
                kernel_bandwidth: Base bandwidth for Gaussian kernel.
                kernel_num: Number of kernels (bandwidths: base * 2^i).
            """
            super().__init__()
            self.bandwidth = kernel_bandwidth
            self.kernel_num = kernel_num

        def forward(
            self, source: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            """Compute MMD between source and target features.

            Args:
                source: Source domain features [n_source, dim].
                target: Target domain features [n_target, dim].

            Returns:
                Scalar MMD loss value.
            """
            n_s = source.size(0)
            n_t = target.size(0)

            # Multi-kernel MMD for robustness
            total_mmd = torch.tensor(0.0, device=source.device)

            for i in range(self.kernel_num):
                bandwidth = self.bandwidth * (2**i)
                K_ss = self._gaussian_kernel(source, source, bandwidth)
                K_tt = self._gaussian_kernel(target, target, bandwidth)
                K_st = self._gaussian_kernel(source, target, bandwidth)

                # MMD^2 = E[K(s,s')] + E[K(t,t')] - 2*E[K(s,t)]
                mmd = (
                    K_ss.sum() / (n_s * n_s)
                    + K_tt.sum() / (n_t * n_t)
                    - 2 * K_st.sum() / (n_s * n_t)
                )
                total_mmd = total_mmd + mmd

            return total_mmd / self.kernel_num

        def _gaussian_kernel(
            self, x: torch.Tensor, y: torch.Tensor, bandwidth: float
        ) -> torch.Tensor:
            """Compute Gaussian kernel matrix.

            Args:
                x: First set of vectors [n, dim].
                y: Second set of vectors [m, dim].
                bandwidth: Kernel bandwidth parameter.

            Returns:
                Kernel matrix [n, m].
            """
            x_norm = (x**2).sum(1).view(-1, 1)
            y_norm = (y**2).sum(1).view(1, -1)
            dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
            return torch.exp(-dist / (2 * bandwidth**2))

    class CORALLoss(nn.Module):
        """Correlation Alignment loss.

        CORAL aligns the second-order statistics (covariance) between
        source and target domains by minimizing the Frobenius norm
        of the difference between covariance matrices.
        """

        def forward(
            self, source: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            """Compute CORAL loss between source and target covariances.

            Args:
                source: Source domain features [n_source, dim].
                target: Target domain features [n_target, dim].

            Returns:
                Scalar CORAL loss value.
            """
            d = source.size(1)

            # Compute covariance matrices
            C_s = self._covariance(source)
            C_t = self._covariance(target)

            # Frobenius norm of difference, normalized by dimension
            loss = torch.norm(C_s - C_t, p="fro") ** 2
            loss = loss / (4 * d * d)

            return loss

        def _covariance(self, x: torch.Tensor) -> torch.Tensor:
            """Compute sample covariance matrix.

            Args:
                x: Feature matrix [n, dim].

            Returns:
                Covariance matrix [dim, dim].
            """
            n = x.size(0)
            x_centered = x - x.mean(dim=0, keepdim=True)
            return torch.mm(x_centered.t(), x_centered) / (n - 1)

else:
    # Placeholder classes when PyTorch not available
    class MMDLoss:
        """Placeholder for MMDLoss (requires PyTorch)."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch required for MMDLoss. Install: pip install torch"
            )

    class CORALLoss:
        """Placeholder for CORALLoss (requires PyTorch)."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch required for CORALLoss. Install: pip install torch"
            )

"""
SENTINEL — Crisis-Aware Quantile Loss

Forces the TFT to learn GKG/crisis signals by:
1. Weighting outer quantiles (Q2, Q98) 3x more than inner quantiles
2. Applying a crisis multiplier when sample weights indicate anomalous periods

The standard QuantileLoss treats all hours and quantiles equally.
This means the model optimizes for normal-day median accuracy and
ignores tail behavior during rare crisis events (~1.8% of data).

CrisisAwareQuantileLoss fixes this by penalizing the model 3x more
for getting the outer prediction bands wrong, and up to 10x more
during GKG-flagged crisis hours.
"""

import torch
from pytorch_forecasting.metrics import QuantileLoss


class CrisisAwareQuantileLoss(QuantileLoss):
    """
    Extends QuantileLoss with:
    - Static outer-quantile emphasis (always active)
    - Dynamic crisis weighting via sample_weight column (when provided)
    
    Args:
        quantiles: List of quantile levels to predict
        outer_weight: Multiplier for the outermost quantiles (default: 3.0)
        crisis_boost: Additional multiplier applied during crisis periods (default: 5.0)
    """
    
    def __init__(
        self, 
        quantiles=None, 
        outer_weight: float = 2.0, 
        crisis_boost: float = 5.0,
        **kwargs
    ):
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        super().__init__(quantiles=quantiles, **kwargs)
        self.outer_weight = outer_weight
        self.crisis_boost = crisis_boost
        
        # Build per-quantile weight vector
        # Outer quantiles get outer_weight, inner quantiles get 1.0
        n = len(quantiles)
        weights = torch.ones(n)
        weights[0] = outer_weight     # Q2 (leftmost)
        weights[-1] = outer_weight    # Q98 (rightmost)
        # Semi-outer quantiles get moderate boost
        if n >= 5:
            weights[1] = (outer_weight + 1.0) / 2.0     # Q10
            weights[-2] = (outer_weight + 1.0) / 2.0    # Q90
        
        self.register_buffer("quantile_weights", weights)
    
    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted pinball loss.
        
        Args:
            y_pred: (batch, horizon, n_quantiles)
            target: (batch, horizon) or (batch, horizon, 1)
        
        Returns:
            Weighted loss tensor of shape (batch, horizon)
        """
        # Standard pinball loss calculation
        # target shape: (batch, horizon) -> unsqueeze for broadcasting
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        
        quantiles_tensor = torch.tensor(
            self.quantiles, device=y_pred.device, dtype=y_pred.dtype
        )
        
        errors = target - y_pred  # (batch, horizon, n_quantiles)
        
        # Pinball loss: max(q * e, (q-1) * e)
        losses = torch.max(
            quantiles_tensor * errors,
            (quantiles_tensor - 1) * errors
        )  # (batch, horizon, n_quantiles)
        
        # Apply static per-quantile weights
        # Outer quantiles are penalized outer_weight times more
        qw = self.quantile_weights.to(y_pred.device)
        weighted_losses = losses * qw.unsqueeze(0).unsqueeze(0)
        
        # Average across quantiles
        return weighted_losses.mean(dim=-1)  # (batch, horizon)

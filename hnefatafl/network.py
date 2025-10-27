"""
Neural Network Architecture for Copenhagen Hnefatafl

ResNet-style CNN with policy and value heads, similar to AlphaZero.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and skip connection"""

    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection"""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual  # Skip connection
        out = F.relu(out)

        return out


class HnefataflNetwork(nn.Module):
    """
    Neural network for Copenhagen Hnefatafl.

    Architecture:
    - Initial convolutional layer
    - N residual blocks
    - Policy head (outputs move probabilities)
    - Value head (outputs position evaluation)

    Input: (batch_size, 15, 11, 11) - 15 channels of 11x11 board state
    Output:
        - Policy: (batch_size, num_possible_moves) - move probabilities
        - Value: (batch_size, 1) - position evaluation in [-1, 1]
    """

    def __init__(self,
                 num_channels: int = 128,
                 num_res_blocks: int = 10,
                 board_size: int = 11):
        super().__init__()

        self.board_size = board_size
        self.num_channels = num_channels

        # Initial conv layer
        self.conv_input = nn.Conv2d(15, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        # For simplicity, we'll use a flat policy output for all from-to moves
        # Max moves: 121 squares * ~20 possible destinations = ~2400 moves
        # We'll use a simpler approach: 121 * 11 * 11 = 14641 possible moves (from any square to any other)
        # But we'll mask illegal moves during MCTS
        self.policy_fc = nn.Linear(32 * board_size * board_size, board_size * board_size * 4 * 10)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 15, 11, 11)

        Returns:
            Tuple of (policy_logits, value)
            - policy_logits: (batch_size, num_actions) - unnormalized move probabilities
            - value: (batch_size, 1) - position evaluation in [-1, 1]
        """
        # Input conv layer
        out = self.conv_input(x)
        out = self.bn_input(out)
        out = F.relu(out)

        # Residual tower
        for res_block in self.res_blocks:
            out = res_block(out)

        # Policy head
        policy = self.policy_conv(out)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy_logits = self.policy_fc(policy)

        # Value head
        value = self.value_conv(out)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)  # Flatten
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)  # Output in [-1, 1]

        return policy_logits, value

    def predict(self, board_state: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Make a prediction for a single board state.

        Args:
            board_state: Tensor of shape (15, 11, 11) or (1, 15, 11, 11)

        Returns:
            Tuple of (policy_probs, value)
            - policy_probs: Tensor of move probabilities
            - value: Float position evaluation in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            if len(board_state.shape) == 3:
                board_state = board_state.unsqueeze(0)

            policy_logits, value = self.forward(board_state)
            policy_probs = F.softmax(policy_logits, dim=1)

            return policy_probs[0], value[0].item()


class PolicyValueLoss(nn.Module):
    """
    Combined loss function for policy and value heads.

    Loss = (z - v)^2 - π^T log p + c||θ||^2
    where:
    - z is the game outcome
    - v is the value prediction
    - π is the MCTS policy
    - p is the network policy
    - c is the L2 regularization parameter
    """

    def __init__(self, value_loss_weight: float = 1.0):
        super().__init__()
        self.value_loss_weight = value_loss_weight

    def forward(self,
                policy_logits: torch.Tensor,
                value_pred: torch.Tensor,
                target_policy: torch.Tensor,
                target_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            policy_logits: Network policy logits (batch_size, num_actions)
            value_pred: Network value predictions (batch_size, 1)
            target_policy: MCTS policy targets (batch_size, num_actions)
            target_value: Game outcome targets (batch_size, 1)

        Returns:
            Tuple of (total_loss, policy_loss, value_loss)
        """
        # Value loss: MSE between predicted value and actual outcome
        value_loss = F.mse_loss(value_pred, target_value)

        # Policy loss: Cross-entropy between MCTS policy and network policy
        policy_loss = -torch.sum(target_policy * F.log_softmax(policy_logits, dim=1)) / policy_logits.size(0)

        # Combined loss
        total_loss = policy_loss + self.value_loss_weight * value_loss

        return total_loss, policy_loss, value_loss


def create_model(num_channels: int = 128,
                 num_res_blocks: int = 10,
                 board_size: int = 11,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> HnefataflNetwork:
    """
    Create and initialize a Hnefatafl network.

    Args:
        num_channels: Number of channels in residual blocks
        num_res_blocks: Number of residual blocks
        board_size: Size of the board (11 for Copenhagen Hnefatafl)
        device: Device to place the model on ('cuda' or 'cpu')

    Returns:
        Initialized HnefataflNetwork
    """
    model = HnefataflNetwork(
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        board_size=board_size
    )

    model = model.to(device)

    # Initialize weights
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    return model


def save_checkpoint(model: HnefataflNetwork,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   path: str):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_channels': model.num_channels,
        'board_size': model.board_size,
    }, path)


def load_checkpoint(path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[HnefataflNetwork, int]:
    """
    Load model checkpoint.

    Returns:
        Tuple of (model, epoch)
    """
    checkpoint = torch.load(path, map_location=device)

    model = create_model(
        num_channels=checkpoint['num_channels'],
        board_size=checkpoint['board_size'],
        device=device
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint['epoch']

# -*- coding: utf-8 -*-
"""
Define a arquitetura da Rede Neural Convolucional (CNN) para o agente DQN.
"""

import torch.nn as nn
import torch.nn.functional as F

# --- Modelo da Rede Neural (DQN com CNN Otimizada) ---
class DQN(nn.Module):
    """
    Rede Neural Convolucional otimizada com Max Pooling para reduzir
    o número de parâmetros e acelerar o treinamento.
    """
    def __init__(self, n_observations_channels, n_actions, max_grid_size=50):
        super().__init__()
        self.conv1 = nn.Conv2d(n_observations_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calcula o tamanho da entrada para a camada linear dinamicamente
        # Após 2 camadas de pooling, o tamanho da imagem (50x50) se torna (12x12)
        # 50 -> 25 -> 12
        linear_input_size = 32 * (max_grid_size // 4) * (max_grid_size // 4)

        self.fc1 = nn.Linear(linear_input_size, n_actions)

    def forward(self, x):
        """Executa a passagem para a frente na rede."""
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # Achata o tensor para a camada totalmente conectada
        x = x.view(x.size(0), -1)
        return self.fc1(x)

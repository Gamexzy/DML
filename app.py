# -*- coding: utf-8 -*-
"""
Servidor Flask para carregar o modelo de IA treinado e resolver
mapas personalizados enviados pelo editor web.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS
import os

# --- Configurações (devem ser as mesmas do train.py) ---
MAX_GRID_SIZE = 50
DEVICE = torch.device("cpu") # O servidor rodará em CPU para compatibilidade
BEST_MODEL_FILENAME = "dqn_escape_room_model_best.pth"
MODEL_FILENAME = "dqn_escape_room_model.pth"

# --- Definição da Arquitetura da Rede Neural (deve ser idêntica à do train.py) ---
class DQN(nn.Module):
    def __init__(self, n_observations_channels, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(n_observations_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * MAX_GRID_SIZE * MAX_GRID_SIZE, n_actions)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

# --- Inicialização do Servidor e do Modelo ---
app = Flask(__name__)
CORS(app)

n_actions = 4
n_observations_channels = 3
policy_net = DQN(n_observations_channels, n_actions).to(DEVICE)

# --- LÓGICA DE CARREGAMENTO CORRIGIDA ---
load_file = None
if os.path.exists(BEST_MODEL_FILENAME):
    load_file = BEST_MODEL_FILENAME
elif os.path.exists(MODEL_FILENAME):
    load_file = MODEL_FILENAME

if load_file:
    print(f"A carregar o melhor modelo de: {load_file}")
    try:
        # Tenta carregar o novo formato (dicionário)
        checkpoint = torch.load(load_file, map_location=DEVICE)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        print("Modelo carregado com sucesso (formato novo).")
    except (KeyError, TypeError):
        # Se falhar, assume que é o formato antigo (apenas os pesos)
        print("Ficheiro de modelo em formato antigo detetado. A carregar apenas os pesos.")
        policy_net.load_state_dict(torch.load(load_file, map_location=DEVICE))
        print("Modelo carregado com sucesso (formato antigo).")
else:
    print(f"ERRO: Nenhum ficheiro de modelo ('{BEST_MODEL_FILENAME}' ou '{MODEL_FILENAME}') foi encontrado.")
    print("Por favor, treine um modelo primeiro usando train.py.")
    exit()

policy_net.eval()
print("Servidor pronto.")
# --- FIM DA CORREÇÃO ---


def get_state_from_map(grid_data, agent_pos, exit_pos):
    """Converte os dados do mapa do editor para um tensor que a IA entende."""
    current_size = len(grid_data)
    state = np.zeros((1, 3, current_size, current_size), dtype=np.float32)
    
    grid_np = np.array(grid_data)
    
    state[0, 0, agent_pos['r'], agent_pos['c']] = 1
    state[0, 1] = grid_np
    state[0, 2, exit_pos['r'], exit_pos['c']] = 1
    
    padded_state = np.ones((1, 3, MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.float32)
    padded_state[:, 0, :, :], padded_state[:, 2, :, :] = 0, 0
    padded_state[:, :, :current_size, :current_size] = state

    return torch.from_numpy(padded_state).to(DEVICE)

@app.route('/solve', methods=['POST'])
def solve_map():
    """Recebe um mapa, usa a IA para encontrar um caminho e retorna a trajetória."""
    data = request.json
    grid = data['grid']
    agent_pos = data['agentPos']
    exit_pos = data['exitPos']
    
    path = [agent_pos]
    max_steps = MAX_GRID_SIZE * MAX_GRID_SIZE

    for _ in range(max_steps):
        state_tensor = get_state_from_map(grid, agent_pos, exit_pos)
        
        with torch.no_grad():
            action = policy_net(state_tensor).max(1)[1].item()

        r, c = agent_pos['r'], agent_pos['c']
        if action == 0: r -= 1
        elif action == 1: r += 1
        elif action == 2: c -= 1
        elif action == 3: c += 1
        agent_pos = {'r': r, 'c': c}
        path.append(agent_pos)

        if agent_pos['r'] == exit_pos['r'] and agent_pos['c'] == exit_pos['c']:
            break
        if grid[agent_pos['r']][agent_pos['c']] == 1:
            break

    return jsonify({'path': path})

if __name__ == '__main__':
    app.run(debug=True)

# -*- coding: utf-8 -*-
"""
Define a classe do ambiente vetorizado para o jogo Escape Room,
onde múltiplos agentes interagem com o mapa simultaneamente.
"""
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Configuração do dispositivo (GPU se disponível, senão CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VectorizedEnvironment:
    """Representa um ambiente com múltiplos agentes e tamanho de mapa variável."""

    def __init__(self, max_size, num_agents, use_proximity_reward=True):
        self.max_size = max_size
        self.num_agents = num_agents
        self.use_proximity_reward = use_proximity_reward

        self.current_size = 0
        self.agent_positions = np.zeros((num_agents, 2), dtype=int)
        self.exit_pos = None
        self.grid = None

        self.agent_colors = [plt.cm.viridis(i / num_agents)[:3] for i in range(num_agents)]
        self.color_map = {0: [1.0, 1.0, 1.0], 1: [0.2, 0.2, 0.2], 2: [0.2, 0.8, 0.2]}
        self.action_deltas = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    def _is_solvable(self, start_node):
        """Verifica se há um caminho do start_node até a saída usando BFS."""
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            r, c = queue.popleft()
            if (r, c) == self.exit_pos:
                return True

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.current_size and 0 <= nc < self.current_size and \
                   self.grid[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def _generate_map_layout(self, size, obstacle_density):
        """Gera a estrutura base do mapa (paredes e obstáculos)."""
        self.current_size = size
        self.grid = np.zeros((size, size), dtype=np.float32)
        # Cria as paredes externas
        self.grid[0, :], self.grid[-1, :], self.grid[:, 0], self.grid[:, -1] = 1, 1, 1, 1

        num_obstacles = int((size ** 2) * obstacle_density)
        valid_positions = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]

        if len(valid_positions) < self.num_agents + num_obstacles + 1:
            return False # Não há espaço suficiente

        obstacle_positions = random.sample(valid_positions, num_obstacles)
        for pos in obstacle_positions:
            self.grid[pos] = 1
        return True

    def _place_entities_and_validate(self):
        """Posiciona a saída e o ponto de spawn, e valida se o mapa é solucionável."""
        valid_positions = [(r, c) for r in range(1, self.current_size - 1) for c in range(1, self.current_size - 1) if self.grid[r, c] == 0]
        if len(valid_positions) < 2:
            return None # Não há espaço para spawn e saída

        random.shuffle(valid_positions)
        spawn_point = valid_positions.pop()
        self.exit_pos = valid_positions.pop()

        if self._is_solvable(spawn_point):
            return spawn_point
        return None

    def reset_map(self, session_index, num_sessions):
        """Gera um novo mapa com dificuldade baseada na sessão (Curriculum Learning)."""
        progress = session_index / num_sessions
        if progress <= 0.2: # Primeiros 20%
            min_size, max_size, obstacle_density = 10, 15, 0.10
        elif progress <= 0.5: # Próximos 30%
            min_size, max_size, obstacle_density = 15, 25, 0.15
        else: # Últimos 50%
            min_size, max_size, obstacle_density = 25, 50, 0.20

        retry_count = 0
        while True:
            if retry_count > 0 and retry_count % 10 == 0:
                print(f"A gerar mapa (dificuldade {progress:.0%})... (tentativa {retry_count})")
            retry_count += 1

            size = random.randint(min_size, max_size)
            if not self._generate_map_layout(size, obstacle_density):
                continue

            spawn_point = self._place_entities_and_validate()
            if spawn_point:
                self.agent_positions[:] = spawn_point
                break

        return self.get_states()

    def get_states(self):
        """Retorna um lote de estados, preenchidos para o tamanho máximo."""
        states = np.zeros((self.num_agents, 3, self.current_size, self.current_size), dtype=np.float32)

        agent_indices = np.arange(self.num_agents)
        states[agent_indices, 0, self.agent_positions[:, 0], self.agent_positions[:, 1]] = 1
        states[:, 1, :, :] = self.grid
        states[:, 2, self.exit_pos[0], self.exit_pos[1]] = 1

        padded_states = np.ones((self.num_agents, 3, self.max_size, self.max_size), dtype=np.float32)
        padded_states[:, 0, :, :], padded_states[:, 2, :, :] = 0, 0
        padded_states[:, :, :self.current_size, :self.current_size] = states

        return torch.from_numpy(padded_states).to(DEVICE)

    def step(self, actions, session_dones):
        """Executa um lote de ações de forma vetorizada."""
        rewards = torch.zeros((self.num_agents, 1), device=DEVICE)
        dones = torch.zeros((self.num_agents, 1), dtype=torch.bool, device=DEVICE)

        active_mask = ~session_dones.cpu().numpy().flatten()
        active_indices = np.where(active_mask)[0]

        if active_indices.size == 0:
            return self.get_states(), rewards, dones, {}

        rewards[active_mask] = -1.0 # Custo de vida por passo

        deltas = self.action_deltas[actions[active_mask].cpu().numpy().flatten()]
        old_positions = self.agent_positions[active_mask]
        new_positions = old_positions + deltas

        won_mask = np.all(new_positions == self.exit_pos, axis=1)
        if np.any(won_mask):
            won_indices = active_indices[won_mask]
            rewards[won_indices] = 100.0
            dones[:] = True # Finaliza a sessão para todos se um agente vencer
            self.agent_positions[won_indices] = new_positions[won_mask]
            return self.get_states(), rewards, dones, {}

        wall_mask = self.grid[new_positions[:, 0], new_positions[:, 1]] == 1
        collided_indices = active_indices[wall_mask]
        rewards[collided_indices] = -20.0
        dones[collided_indices] = True

        moved_mask = ~wall_mask
        moved_indices = active_indices[moved_mask]
        if self.use_proximity_reward and moved_indices.size > 0:
            old_dist = np.abs(old_positions[moved_mask] - self.exit_pos).sum(axis=1)
            new_dist = np.abs(new_positions[moved_mask] - self.exit_pos).sum(axis=1)

            closer_mask = new_dist < old_dist
            further_mask = ~closer_mask

            rewards[moved_indices[closer_mask]] += 0.5
            rewards[moved_indices[further_mask]] += -0.6

        self.agent_positions[moved_indices] = new_positions[moved_mask]

        return self.get_states(), rewards, dones, {}

    def render(self, ax, session, step, session_dones, q_values=None):
        """Renderiza o estado atual com todos os agentes e o heatmap de Q-values."""
        display_grid = np.copy(self.grid)
        display_grid[self.exit_pos] = 2

        img = np.array([self.color_map[val] for val in display_grid.flatten()]).reshape(self.current_size, self.current_size, 3)

        if q_values is not None and not session_dones[0]:
            probs = F.softmax(q_values, dim=1).squeeze().cpu().numpy()

            agent0_pos = self.agent_positions[0]
            for i, (dr, dc) in enumerate(self.action_deltas):
                r, c = agent0_pos[0] + dr, agent0_pos[1] + dc
                if 0 <= r < self.current_size and 0 <= c < self.current_size:
                    color = np.array([0.8, 0.2, 0.8]) # Cor roxa para heatmap
                    alpha = probs[i] * 0.9
                    img[r, c] = img[r, c] * (1 - alpha) + color * alpha

        for i, pos in enumerate(self.agent_positions):
            alpha = 1.0 if not session_dones[i] else 0.3
            color = self.agent_colors[i]
            img[pos[0], pos[1]] = img[pos[0], pos[1]] * (1 - alpha) + np.array(color) * alpha

        ax.clear()
        ax.imshow(img)
        ax.set_title(f"Sessão: {session} | Passo: {step} | Tamanho: {self.current_size}x{self.current_size}")
        ax.set_xticks([])
        ax.set_yticks([])

# -*- coding: utf-8 -*-
"""
Script para treinar uma IA para escapar de mapas com tamanhos variados,
usando uma estratégia de Curriculum Learning e Aprendizagem Coletiva Contínua,
onde a IA aprende com a experiência de todos os agentes.
"""

# Imports principais
import math
import os
import random
import sys
from collections import deque, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- Configurações e Hiperparâmetros ---

# ATIVE/DESATIVE A VISUALIZAÇÃO AQUI
VISUALIZE = True
VISUALIZATION_DELAY = 0.05

# --- LÓGICA DE RECOMPENSA POR PROXIMIDADE ---
USE_PROXIMITY_REWARD = True
PROXIMITY_REWARD_BONUS = 0.5
PROXIMITY_REWARD_PENALTY = -0.6
# -----------------------------------------

# Configuração de salvamento
CHECKPOINT_FREQ = 25
MODEL_FILENAME = "dqn_escape_room_model.pth"
BEST_MODEL_FILENAME = "dqn_escape_room_model_best.pth"

# Configuração do dispositivo (GPU se disponível, senão CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parâmetros do Ambiente
MAX_GRID_SIZE = 50

# Parâmetros do Treinamento
NUM_SESSIONS = 500
NUM_AGENTS = 10
MAX_STEPS_PER_SESSION = 300
INITIAL_RANDOM_STEPS = 10

BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50000
TAU = 0.005
LR = 1e-4

# Definição de uma "Transição"
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)


# --- Classe da Memória de Replay ---
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)


# --- Classe do Ambiente Vetorizado ---
class VectorizedEnvironment:
    """Representa um ambiente com múltiplos agentes e tamanho de mapa variável."""

    def __init__(self, max_size, num_agents):
        self.max_size = max_size
        self.num_agents = num_agents
        
        self.current_size = 0
        self.agent_positions = np.zeros((num_agents, 2), dtype=int)
        self.exit_pos = None
        self.grid = None
        
        self.agent_colors = [plt.cm.viridis(i / num_agents)[:3] for i in range(num_agents)]
        self.color_map = {
            0: [1.0, 1.0, 1.0], 1: [0.2, 0.2, 0.2], 2: [0.2, 0.8, 0.2]
        }
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

    def reset_map(self, session_index):
        """Gera um novo mapa com dificuldade baseada na sessão (Curriculum Learning)."""
        if session_index <= 100:
            min_size, max_size = 10, 15
            obstacle_density = 0.10
        elif session_index <= 250:
            min_size, max_size = 15, 25
            obstacle_density = 0.15
        else:
            min_size, max_size = 25, 50
            obstacle_density = 0.20

        retry_count = 0
        while True:
            if retry_count > 0 and retry_count % 10 == 0:
                print(f"A gerar mapa (dificuldade {session_index / NUM_SESSIONS:.0%})... (tentativa {retry_count})")
            retry_count += 1

            self.current_size = random.randint(min_size, max_size)
            num_obstacles = int((self.current_size ** 2) * obstacle_density)
            
            self.grid = np.zeros((self.current_size, self.current_size), dtype=np.float32)
            self.grid[0, :], self.grid[-1, :], self.grid[:, 0], self.grid[:, -1] = 1, 1, 1, 1
            
            valid_positions = [(r, c) for r in range(1, self.current_size - 1) for c in range(1, self.current_size - 1)]
            
            if len(valid_positions) < self.num_agents + num_obstacles + 1:
                continue

            random.shuffle(valid_positions)
            
            spawn_point = valid_positions.pop()
            self.exit_pos = valid_positions.pop()
            
            obstacle_positions = random.sample(valid_positions, num_obstacles)
            for pos in obstacle_positions:
                self.grid[pos] = 1
            
            if self._is_solvable(spawn_point):
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

        rewards[active_mask] = -1.0
        
        deltas = self.action_deltas[actions[active_mask].cpu().numpy().flatten()]
        old_positions = self.agent_positions[active_mask]
        new_positions = old_positions + deltas

        won_mask = np.all(new_positions == self.exit_pos, axis=1)
        if np.any(won_mask):
            won_indices = active_indices[won_mask]
            rewards[won_indices] = 100.0
            dones[:] = True
            self.agent_positions[won_indices] = new_positions[won_mask]
            return self.get_states(), rewards, dones, {}

        wall_mask = self.grid[new_positions[:, 0], new_positions[:, 1]] == 1
        collided_indices = active_indices[wall_mask]
        rewards[collided_indices] = -20.0 # Penalidade ajustada
        dones[collided_indices] = True

        moved_mask = ~wall_mask
        moved_indices = active_indices[moved_mask]
        if USE_PROXIMITY_REWARD and moved_indices.size > 0:
            old_dist = np.abs(old_positions[moved_mask] - self.exit_pos).sum(axis=1)
            new_dist = np.abs(new_positions[moved_mask] - self.exit_pos).sum(axis=1)
            
            closer_mask = new_dist < old_dist
            further_mask = ~closer_mask

            rewards[moved_indices[closer_mask]] += PROXIMITY_REWARD_BONUS
            rewards[moved_indices[further_mask]] += PROXIMITY_REWARD_PENALTY

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
                    color = np.array([0.8, 0.2, 0.8])
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
        plt.pause(VISUALIZATION_DELAY)


# --- Modelo da Rede Neural (DQN com CNN) ---
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


# --- Funções Auxiliares de Treinamento ---
def select_actions(states, policy_net, current_step_in_session):
    """Seleciona um lote de ações. Força exploração no início da sessão."""
    global steps_done
    steps_done += 1
    
    if current_step_in_session < INITIAL_RANDOM_STEPS:
        return torch.tensor([[random.randrange(n_actions)] for _ in range(NUM_AGENTS)], device=DEVICE, dtype=torch.long)

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(states).max(1)[1].view(NUM_AGENTS, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)] for _ in range(NUM_AGENTS)], device=DEVICE, dtype=torch.long)

def plot_progress(ax, show_result=False):
    ax.clear()
    rewards_t = torch.tensor(session_avg_rewards, dtype=torch.float)
    if show_result: ax.set_title("Resultado Final")
    else: ax.set_title("Progresso do Treinamento")
    ax.set_xlabel("Sessão")
    ax.set_ylabel("Recompensa Média da Sessão")
    ax.plot(rewards_t.numpy(), label="Recompensa Média")
    if len(rewards_t) >= 50:
        means = rewards_t.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        ax.plot(means.numpy(), label="Média (50 sessões)")
    ax.legend()

def optimize_model():
    if len(memory) < BATCH_SIZE: return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch, action_batch, reward_batch = torch.cat(batch.state), torch.cat(batch.action), torch.cat(batch.reward)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# --- Inicialização ---
env = VectorizedEnvironment(MAX_GRID_SIZE, NUM_AGENTS)
n_actions = 4
state = env.reset_map(1)
n_observations_channels = state.shape[1]

policy_net = DQN(n_observations_channels, n_actions).to(DEVICE)
target_net = DQN(n_observations_channels, n_actions).to(DEVICE)

best_avg_reward = -float('inf')

load_file = None
if os.path.exists(BEST_MODEL_FILENAME):
    load_file = BEST_MODEL_FILENAME
elif os.path.exists(MODEL_FILENAME):
    load_file = MODEL_FILENAME

if load_file:
    print(f"Carregando modelo existente de: {load_file}")
    try:
        checkpoint = torch.load(load_file)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
        print(f"Nível do melhor agente carregado: {best_avg_reward:.2f}")
    except (KeyError, TypeError, RuntimeError) as e:
        print(f"A carregar um modelo de formato antigo ou incompatível. Apenas os pesos serão carregados. Erro: {e}")
        policy_net.load_state_dict(torch.load(load_file, weights_only=True))
        best_avg_reward = -float('inf')

else:
    print("Nenhum modelo encontrado. Iniciando um novo treino.")

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(40000)
steps_done = 0
session_avg_rewards = []

# --- Loop Principal de Treinamento ---
if __name__ == "__main__":
    print(f"Iniciando treino para {NUM_SESSIONS} sessões com {NUM_AGENTS} agentes em paralelo.")
    if VISUALIZE:
        plt.ion()
        fig_env, ax_env = plt.subplots(figsize=(8, 8))
        fig_plot, ax_plot = plt.subplots(figsize=(8, 5))
    
    try:
        for i_session in range(1, NUM_SESSIONS + 1):
            states = env.reset_map(i_session)
            session_dones = torch.zeros((NUM_AGENTS, 1), dtype=torch.bool, device=DEVICE)
            session_total_reward = 0

            for t in range(MAX_STEPS_PER_SESSION):
                actions = select_actions(states, policy_net, t)
                next_states, rewards, dones_this_step, _ = env.step(actions, session_dones)
                
                active_agents_mask = ~session_dones.flatten()
                if active_agents_mask.any():
                    session_total_reward += rewards[active_agents_mask].mean().item()

                for i in range(NUM_AGENTS):
                    if not session_dones[i]:
                        is_done_now = dones_this_step[i]
                        next_state_i = next_states[i].unsqueeze(0) if not is_done_now else None
                        memory.push(states[i].unsqueeze(0), actions[i].view(1,1), next_state_i, rewards[i].view(1))

                states = next_states
                session_dones |= dones_this_step
                
                optimize_model()

                if VISUALIZE:
                    with torch.no_grad():
                        q_values_agent0 = policy_net(states[0].unsqueeze(0))
                    env.render(ax_env, i_session, t + 1, session_dones.cpu().numpy(), q_values_agent0)
                else:
                    if t % 10 == 0:
                        print(f"\rSessão {i_session}, Passo {t}/{MAX_STEPS_PER_SESSION}", end="")

                if session_dones.all():
                    break
            
            num_steps = t + 1
            session_avg_rewards.append(session_total_reward / num_steps)
            avg_reward_overall = np.mean(session_avg_rewards[-50:])

            if i_session % 10 == 0:
                print(f"\nSessão {i_session}: Recompensa Média: {session_avg_rewards[-1]:.2f}, Média Geral (50 sessões): {avg_reward_overall:.2f}")

            if len(session_avg_rewards) >= 50 and avg_reward_overall > best_avg_reward:
                best_avg_reward = avg_reward_overall
                torch.save({
                    'model_state_dict': policy_net.state_dict(),
                    'best_avg_reward': best_avg_reward,
                }, BEST_MODEL_FILENAME)
                print(f"*** Novo melhor modelo salvo! Nível: {best_avg_reward:.2f} ***")

            if i_session % CHECKPOINT_FREQ == 0:
                torch.save({
                    'model_state_dict': policy_net.state_dict(),
                    'best_avg_reward': best_avg_reward,
                }, MODEL_FILENAME)
                print(f"--- Checkpoint salvo em {MODEL_FILENAME} (Sessão {i_session}) ---")

            if VISUALIZE:
                plot_progress(ax_plot)
                fig_plot.canvas.draw()
                fig_plot.canvas.flush_events()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nTreino interrompido.")
    finally:
        print("\nSalvando modelo final...")
        torch.save({
            'model_state_dict': policy_net.state_dict(),
            'best_avg_reward': best_avg_reward,
        }, MODEL_FILENAME)
        print(f"Modelo salvo em {MODEL_FILENAME}")
        plt.ioff()
        if session_avg_rewards:
            final_fig, final_ax = plt.subplots(figsize=(8, 8))
            plot_progress(final_ax, show_result=True)
            plt.show()
        print("Treino finalizado.")
        sys.exit(0)

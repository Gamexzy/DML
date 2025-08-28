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
import torch.optim as optim
from environment import VectorizedEnvironment
from model import DQN

# --- Configurações e Hiperparâmetros ---

# ATIVE/DESATIVE A VISUALIZAÇÃO AQUI
VISUALIZE = True
VISUALIZATION_DELAY = 0.01

# --- LÓGICA DE RECOMPENSA POR PROXIMIDADE ---
USE_PROXIMITY_REWARD = True

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
    """Armazena as transições observadas pelos agentes."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)


# --- Funções Auxiliares de Treinamento ---
def select_actions(states, policy_net, current_step_in_session, n_actions):
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

def plot_progress(ax, session_rewards, show_result=False):
    """Plota o gráfico de progresso do treinamento."""
    ax.clear()
    rewards_t = torch.tensor(session_rewards, dtype=torch.float)
    if show_result:
        ax.set_title("Resultado Final")
    else:
        ax.set_title("Progresso do Treinamento")
    ax.set_xlabel("Sessão")
    ax.set_ylabel("Recompensa Média da Sessão")
    ax.plot(rewards_t.numpy(), label="Recompensa Média")
    if len(rewards_t) >= 50:
        means = rewards_t.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        ax.plot(means.numpy(), label="Média (50 sessões)")
    ax.legend()
    plt.pause(0.001)

def optimize_model(memory, policy_net, target_net, optimizer):
    """Otimiza o modelo da rede neural."""
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

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


# --- Loop Principal de Treinamento ---
def main():
    """Função principal que executa o loop de treinamento."""
    global steps_done
    steps_done = 0

    env = VectorizedEnvironment(MAX_GRID_SIZE, NUM_AGENTS, USE_PROXIMITY_REWARD)
    n_actions = 4
    # Reset inicial para obter o número de canais de observação
    initial_state = env.reset_map(1, NUM_SESSIONS)
    n_observations_channels = initial_state.shape[1]

    policy_net = DQN(n_observations_channels, n_actions, MAX_GRID_SIZE).to(DEVICE)
    target_net = DQN(n_observations_channels, n_actions, MAX_GRID_SIZE).to(DEVICE)

    best_avg_reward = -float('inf')

    # Lógica para carregar um modelo pré-existente
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
            print(f"A carregar um modelo de formato antigo ou incompatível. Erro: {e}")
            policy_net.load_state_dict(torch.load(load_file, map_location=DEVICE))
            best_avg_reward = -float('inf')
    else:
        print("Nenhum modelo encontrado. Iniciando um novo treino.")

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(40000)
    session_avg_rewards = []

    print(f"Iniciando treino para {NUM_SESSIONS} sessões com {NUM_AGENTS} agentes em paralelo em {DEVICE}.")
    if VISUALIZE:
        plt.ion()
        fig_env, ax_env = plt.subplots(figsize=(8, 8))
        fig_plot, ax_plot = plt.subplots(figsize=(8, 5))

    try:
        for i_session in range(1, NUM_SESSIONS + 1):
            states = env.reset_map(i_session, NUM_SESSIONS)
            session_dones = torch.zeros((NUM_AGENTS, 1), dtype=torch.bool, device=DEVICE)
            session_total_reward = 0

            for t in range(MAX_STEPS_PER_SESSION):
                actions = select_actions(states, policy_net, t, n_actions)
                next_states, rewards, dones_this_step, _ = env.step(actions, session_dones)

                active_agents_mask = ~session_dones.flatten()
                if active_agents_mask.any():
                    session_total_reward += rewards[active_agents_mask].mean().item()

                for i in range(NUM_AGENTS):
                    if not session_dones[i]:
                        is_done_now = dones_this_step[i]
                        next_state_i = next_states[i].unsqueeze(0) if not is_done_now else None
                        memory.push(states[i].unsqueeze(0), actions[i].view(1, 1), next_state_i, rewards[i].view(1))

                states = next_states
                session_dones |= dones_this_step

                optimize_model(memory, policy_net, target_net, optimizer)

                if VISUALIZE:
                    with torch.no_grad():
                        q_values_agent0 = policy_net(states[0].unsqueeze(0))
                    env.render(ax_env, i_session, t + 1, session_dones.cpu().numpy(), q_values_agent0)
                    plt.pause(VISUALIZATION_DELAY)
                elif t % 10 == 0:
                    print(f"\rSessão {i_session}, Passo {t}/{MAX_STEPS_PER_SESSION}", end="")

                if session_dones.all():
                    break

            # Soft update da rede alvo
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            num_steps = t + 1
            session_avg_rewards.append(session_total_reward / num_steps)
            avg_reward_overall = np.mean(session_avg_rewards[-50:])

            if i_session % 10 == 0:
                print(f"\nSessão {i_session}: Recompensa Média: {session_avg_rewards[-1]:.2f}, Média Geral (50 sessões): {avg_reward_overall:.2f}")

            if len(session_avg_rewards) >= 50 and avg_reward_overall > best_avg_reward:
                best_avg_reward = avg_reward_overall
                torch.save({'model_state_dict': policy_net.state_dict(), 'best_avg_reward': best_avg_reward}, BEST_MODEL_FILENAME)
                print(f"*** Novo melhor modelo salvo! Nível: {best_avg_reward:.2f} ***")

            if i_session % CHECKPOINT_FREQ == 0:
                torch.save({'model_state_dict': policy_net.state_dict(), 'best_avg_reward': best_avg_reward}, MODEL_FILENAME)
                print(f"--- Checkpoint salvo em {MODEL_FILENAME} (Sessão {i_session}) ---")

            if VISUALIZE:
                plot_progress(ax_plot, session_avg_rewards)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nTreino interrompido.")
    finally:
        print("\nSalvando modelo final...")
        torch.save({'model_state_dict': policy_net.state_dict(), 'best_avg_reward': best_avg_reward}, MODEL_FILENAME)
        print(f"Modelo salvo em {MODEL_FILENAME}")
        if VISUALIZE:
            plt.ioff()
        if session_avg_rewards:
            final_fig, final_ax = plt.subplots(figsize=(8, 8))
            plot_progress(final_ax, session_avg_rewards, show_result=True)
            plt.show()
        print("Treino finalizado.")

if __name__ == "__main__":
    main()

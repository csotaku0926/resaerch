# 初始化環境與網路
env = SatelliteDataDisseminationEnv(num_leos=10, num_neighbors=2, num_grids=1)
actors = {agent: LocalActor(obs_dim, action_dim) for agent in env.possible_agents}
# 共用一個強大的中央 Critic，或每個 Agent 一個中央 Critic 皆可
central_critic = CentralizedCritic(global_state_dim)

optimizer_actor = torch.optim.Adam([p for a in actors.values() for p in a.parameters()], lr=1e-4)
optimizer_critic = torch.optim.Adam(central_critic.parameters(), lr=1e-3)

for episode in range(MAX_EPISODES):
    local_obs_dict, _ = env.reset()
    
    # 儲存軌跡 (Trajectories)
    batch_obs, batch_states, batch_actions, batch_rewards = {a:[] for a in env.agents}, [], {a:[] for a in env.agents}, {a:[] for a in env.agents}
    
    # --- 階段一：分散執行 (Decentralized Execution - 收集資料) ---
    while env.agents:
        # 從環境取得上帝視角狀態 (準備給 Critic 用)
        global_state = env.state() 
        batch_states.append(global_state)
        
        actions_dict = {}
        for agent in env.agents:
            # Actor 只根據自己的局部視野做決策
            obs_tensor = torch.FloatTensor(local_obs_dict[agent])
            action_prob = actors[agent](obs_tensor)
            actions_dict[agent] = action_prob.detach().numpy()
            
            batch_obs[agent].append(local_obs_dict[agent])
            batch_actions[agent].append(actions_dict[agent])
            
        # 與環境互動
        next_local_obs, local_rewards, terminations, truncations, infos = env.step(actions_dict)
        
        for agent in env.agents:
            batch_rewards[agent].append(local_rewards[agent])
            
        local_obs_dict = next_local_obs
        
        if all(terminations.values()):
            break

    # --- 階段二：集中訓練 (Centralized Training - 更新權重) ---
    state_tensor = torch.FloatTensor(batch_states) # [Time, Global_Dim]
    
    # 1. 計算全局 V 值 (上帝視角的評價)
    V_values = central_critic(state_tensor).squeeze()
    
    # 2. 針對每個 Agent 計算 Advantage 並更新 Actor
    for agent in env.possible_agents:
        reward_tensor = torch.FloatTensor(batch_rewards[agent])
        # 計算優勢函數 Advantage (實際 Reward - Critic預期的 V)
        advantage = reward_tensor - V_values.detach()[:len(reward_tensor)]
        
        # Actor Loss (PPO 簡化版): -log_prob * advantage
        obs_tensor = torch.FloatTensor(batch_obs[agent])
        action_preds = actors[agent](obs_tensor)
        actor_loss = -torch.mean(action_preds * advantage.unsqueeze(1)) 
        
        optimizer_actor.zero_grad()
        actor_loss.backward()
        optimizer_actor.step()
        
    # 3. 更新 Centralized Critic (讓它評估得更準)
    # Critic 的目標是逼近實際獲得的 Discounted Rewards
    # (此處省略 Gt 的計算，用 reward_tensor 示意)
    target_V = ... # 計算 Return (Gt)
    critic_loss = nn.MSELoss()(V_values, target_V)
    
    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()
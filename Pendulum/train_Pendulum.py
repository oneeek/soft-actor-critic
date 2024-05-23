#%%
import Agent

import gym
import numpy as np
import tensorflow as tf



def train(agent, max_epi_num):
    epi_reward = []
    agent.update_target_network(1.0)

    for ep in range(int(max_epi_num)):
        time, episode_reward, done = 0, 0, False
        state, _ = env.reset()

        while not done:
            action = agent.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
            action = np.clip(action, -agent.action_bound, agent.action_bound)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.buffer.add_buffer(state, action, reward, next_state, done)

            if agent.buffer.buffer_count() > 1000:
                states, actions, rewards, next_states, dones = agent.buffer.sample_batch(agent.BATCH_SIZE)

                next_mu, next_std = agent.actor(tf.convert_to_tensor(next_states, dtype=tf.float32))
                next_actions, next_log_pdf = agent.actor.sample_normal(next_mu, next_std)

                target_qs = agent.target_critic([next_states, next_actions])
                target_qi = target_qs - agent.ALPHA * next_log_pdf
                y_i = agent.q_target(rewards, target_qi.numpy(), dones)

                agent.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                  tf.convert_to_tensor(actions, dtype=tf.float32),
                                  tf.convert_to_tensor(y_i, dtype=tf.float32))
                agent.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32))
                agent.update_target_network(agent.TAU)

            state = next_state
            episode_reward += reward
            time += 1

        print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

        epi_reward.append(episode_reward)
    return epi_reward



#%%
##### env, agent 정의 #####
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

agent = Agent.SACagent(state_dim, action_dim, action_bound)

#%%
##### 학습 #####
max_epi_num = 100
epi_reward = train(agent, max_epi_num)

#%%
import os
import matplotlib.pyplot as plt



##### epi_reward plot #####
plt.plot(epi_reward)
plt.show()

##### save weights #####
cur_dir = os.getcwd()
ckpt_dir = 'checkpoint'
dr = os.path.join(cur_dir, ckpt_dir)
os.makedirs(dr, exist_ok=True)

file_name_a = 'actor_w'
file_name_c = 'critic_w'
file_path_a = os.path.join(dr, file_name_a)
file_path_c = os.path.join(dr, file_name_c)

agent.actor.save_weights(file_path_a)
agent.critic.save_weights(file_path_c)

#%%
##### load weights #####
env = gym.make('Pendulum-v1', render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

agent = Agent.SACagent(state_dim, action_dim, action_bound)



agent.actor.load_weights(file_path_a)
agent.critic.load_weights(file_path_c)

##### test #####
time, done = 0, False
state, _ = env.reset()

while not done:
    action = agent.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
    state, reward, term, trunc, _ = env.step(action)
    done = term or trunc
    time += 1

    print('Time: ', time, 'Reward: ', reward)

env.close()

# %%

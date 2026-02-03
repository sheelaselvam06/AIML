from turtle import right
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
 
print(gym.envs.registry.keys())
env=gym.make("FrozenLake-v1",is_slippery=False, render_mode="rgb_array") #slipper = true slip to l,r,u,d based on the 1/3rd of the probability
env.reset()
rgb_array = env.render()
plt.imshow(rgb_array)
plt.savefig('frozen_lake.png')
print(env.observation_space.n)
print(env.action_space.n)
print(env.step(2)) #right -2
print(env.step(2)) #right
print(env.step(1)) #down
print(env.step(1)) #down
print(env.step(1)) #down
print(env.step(2)) #right
 
env.reset() #reset to imitial state
act = env.action_space.sample() #random action
print(env.step(act))
 
 
 
env2=gym.make("FrozenLake-v1",is_slippery=True, render_mode="rgb_array") #slipper = true slip to l,r,u,d based on the 1/3rd of the probability
env2.reset()
print(env2.step(2)) #right
print(env2.step(2)) #right
print(env2.step(2)) #right
 
env2.reset()
 
print(env2.unwrapped.P[14][2]) #it gives posible moves when turn right->slipper->90deg  
#s[(0.33333333333333337, 14, 0, False)] = [prob,next_state,reward,done]
"""
    #prob wrap rewards nd shape then into values
    #two types of values
    1.state value
    2.state nd action combined val
"""
prob,next_state,reward,done=env2.unwrapped.P[14][2][1]
for prob,next_state,reward,done in env2.unwrapped.P[14][2]:
    print(f"Probability: {prob}, Next State: {next_state}, Reward: {reward}, Done: {done}")
 
#array of max val
Vmax=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] #assumption for nxt state rewards
 
values=0 #reward -> shaped in such way [mul with prob]
for prob,next_state,reward,done in env2.unwrapped.P[14][2]:
    values+=prob*(reward + 0.9 *Vmax[next_state])
    print(values)
 
ls=[] #list to store values to get max value
for a in range(4):
    values=0 #reward -> shaped in such way [mul with prob]
    for prob,next_state,reward,done in env2.unwrapped.P[14][a]:
        values+=prob*(reward + 0.9 *Vmax[next_state]) #reward ->nxt_state reward; vmax[nxt_st]->old val
    ls.append(values)
    #print(f"At position 14, taking action {a}, the value is:", values) #cumulatively added the previous val nd get end result
    #print("__________________________________________________________________________________")
 
print("The maximum value for 14th position is:", max(ls))
Vmax[14] = max(ls)
print("The updated Vmax array is:", Vmax)
 
#14 -> nxt state 18,10,15 -> suppose 15 max val then it get developed
#if it is stay at 14 then we get new loops
#we need to move loop till the val get dimnished
#idea is to get vmax to reward so that it cant move further
 
 
#for state
Vmax=[0.0]*16
for s in range(16):
    ls=[]
    for a in range(4):
        values=0
        for prob,next_state,reward,done in env2.unwrapped.P[s][a]:
            values+=prob*(reward + 0.9 *Vmax[next_state])
        ls.append(values)
    Vmax[s] = max(ls)
 
print("The updated Vmax array is:", Vmax)
 
 
#for multiple iterations
Vmax=[0.0]*16
i = 100
for iteration in range(i):
    for s in range(16):
        ls=[]
        for a in range(4):
            values=0
            for prob,next_state,reward,done in env2.unwrapped.P[s][a]:
                values+=prob*(reward + 0.9 *Vmax[next_state])
            ls.append(values)
        Vmax[s] = max(ls)
print("The updated Vmax array is:", Vmax)
 
"""
#diff b/w present nd previous vmax
#then set the threshold val
Vmax=[0.0]*16
threshold = 1e-8
i = 100
 
for iteration in range(i):
    Vmax_old = Vmax.copy()
    # Store previous values
    for s in range(16):
        ls=[]
        for a in range(4):
            values=0
            for prob,next_state,reward,done in env2.unwrapped.P[s][a]:
                values+=prob*(reward + 0.9 *Vmax[next_state])
            ls.append(values)
        Vmax[s] = max(ls)
   
    # Calculate maximum difference between current and previous Vmax
    max_diff = max(abs(Vmax[s] - Vmax_old[s]) for s in range(16))
    sum_diff = sum(abs(Vmax[s] - Vmax_old[s]) for s in range(16))
    print(f"Iteration {iteration + 1}: Max difference = {max_diff:.6f}, Sum difference = {sum_diff:.6f}")
   
    # Check if converged (difference below threshold)
    if max_diff < threshold:
        print(f"Converged after {iteration + 1} iterations!")
        break
 
print("Final Vmax array:", [f"{v:.6f}" for v in Vmax])
"""
env2.reset()
#another way
Vmax=[0.0]*16
policy=[0]*16  # Store best action for each state
threshold = 1e-8
i = 1000
 
for iteration in range(i):
    # Store previous values
    delta=0
    for s in range(16):
        ls=[]
        for a in range(4):
            values=0
            for prob,next_state,reward,done in env2.unwrapped.P[s][a]:
                values+=prob*(reward + 0.9 *Vmax[next_state])
            ls.append(values)
   
        # Calculate maximum difference between current and previous Vmax
        best_action_value=max(ls)
        delta=max(delta,abs(Vmax[s]-best_action_value))
        Vmax[s]=best_action_value
   
    # Check if converged (difference below threshold) - AFTER all states updated
    if delta < threshold:
        print(f"Converged after {iteration + 1} iterations!")
        break
 
# Use argmax on Vmax to get best state, and sum for total value
best_state = np.argmax(Vmax)
total_value = sum(Vmax)
print("Final Vmax array:", [f"{v:.6f}" for v in Vmax])
print("Best state (argmax of Vmax):", best_state)
print("Total value (sum of Vmax):", total_value)
 
#to find optimum root is called policy
 
 
#derive policy from value function
env2.reset()
policy=[0]*16
for s in range(16):
    ls=[]
    for a in range(4):
        values=0
        for prob,next_state,reward,done in env2.unwrapped.P[s][a]:
            values+=prob*(reward + 0.9 *Vmax[next_state])
        ls.append(values)
    policy[s] = np.argmax(ls)
   
print("Final policy:", policy)
print(np.array(policy).reshape(4,4))
state,_=env2.reset()
env2.render()
plt.savefig("frozen_lake.png")
done=False
while not done:
    action=policy[state]
    state,reward,done,truncated,info=env2.step(action)
    env2.render()
if reward == 1:
    print("Success!")
else:
    print("Failed!")
 
env2.reset()
 
# Policy Iteration Setup
num_states = 16  # Number of states in FrozenLake
num_actions = 4  # Number of actions (Left, Down, Right, Up)
policy = np.random.choice(num_actions, size=num_states)  # Random initial policy
V = np.zeros(num_states)  # Initialize value function
gamma = 0.9  # Discount factor
theta = 1e-6  # Less strict threshold - more iterations
max_iterations = 1000
 
#policy iteration
for i in range(max_iterations):
    #policy evaluation
    while True:
        delta=0
        for s in range(num_states):
            a=policy[s]
            value=0
            for prob,next_state,reward,done in env2.unwrapped.P[s][a]:
                value+=prob*(reward + gamma*V[next_state])
            delta=max(delta,abs(V[s]-value))
            V[s]=value
       
        if delta<theta:
            break
    #policy improvement
    #Q -> combination of s & v
    #state val-> max(v) that is only 1 val [dont have action val]
    policy_stable=True
    for s in range(num_states):
        ls=[]
        for a in range(num_actions):
            value=0
            for prob,next_state,reward,done in env2.unwrapped.P[s][a]:
                value+=prob*(reward + gamma*V[next_state])
            ls.append(value)
        best_action_value=np.argmax(ls)
        if policy[s] != best_action_value:
            policy_stable=False
        policy[s]=best_action_value
       
    if policy_stable:
        print(f"Policy converged after {i+1} iterations")
        break
 
print("Final policy:", policy)
print(np.array(policy).reshape(4,4))
state,_=env2.reset()
env2.render()
plt.savefig("frozen_lake.png")
done=False
while not done:
    action=policy[state]
    state,reward,done,truncated,info=env2.step(action)
    env2.render()
if reward == 1:
    print("Success!")
else:
    print("Failed!")
 
env2.reset()  
 
snv=gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
action=snv.reset()
 
 
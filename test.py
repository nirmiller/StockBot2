batch_size = 32
sell_option = 0

state = getState(sell_option, 0, TIME_RANGE, PRICE_RANGE)

for t in range(0, l):

    action = agent.act(state)

    if t % 2 != 0:
        sell_option = 1
    else:
        sell_option = 0

    # sit
    next_state = getState(sell_option, t + 1, TIME_RANGE, PRICE_RANGE)
    # print(next_state)
    reward = 0

    if action == 0:  # buy
        if sell_option != 0:
            reward = -1
        else:
            reward = 1
            correct += 1
    elif action == 1:  # sell
        if sell_option != 1:
            reward = -1
        else:
            reward = 1
            correct += 1

    print(f"Reward : {reward}")
    agent.total_inventory.append(agent.inventory)

    done = True if t == l - 1 else False
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state

    if done:
        print(e)
        print("--------------------------------")

print(correct / l)

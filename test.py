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
elif action == 1:  # sell
    if sell_option != 1:
        reward = -1
    else:
        reward = 1

print(f"Reward : {reward}")
agent.total_inventory.append(agent.inventory)

done = True if t == l - 1 else False
agent.memory.append((state, action, reward, next_state, done))
state = next_state

if done:
    print(e)
    print("--------------------------------")

if len(agent.memory) > batch_size:
    agent.expReplay(batch_size)
    print("REPLAY {}".format(agent.epsilon))

if e % 5 == 0:
    agent.model.save("/content/drive/MyDrive/StockBot/models/stock_bot_comp/test/test_{}".format(str(e)))

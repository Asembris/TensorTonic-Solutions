def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Perform one SARSA update and return the updated Q-table.
    """
    old_value=q_table[state][action]
    next_value=q_table[next_state][next_action]
    q_table[state][action]=(1-alpha)*old_value+alpha*(reward+gamma*next_value)
    return q_table
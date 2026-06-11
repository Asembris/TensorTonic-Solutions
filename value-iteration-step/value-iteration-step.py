def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    num_states=len(values)
    num_actions=len(rewards[0])
    def get_max_action(state):
        action_scores=[rewards[state][a]+gamma*sum([transitions[state][a][new_state] * values[new_state] for new_state in range(num_states)]) for a in range(num_actions)]
        return max(action_scores)

    new_v=[get_max_action(state) for state in range(num_states)]
    return new_v
 
    
    
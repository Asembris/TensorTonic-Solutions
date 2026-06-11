def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    # Write code here
    t=len(rewards)
    dis_returns=[0]*t
    dis_returns[-1]=rewards[-1]
    for r in range(t-2,-1,-1):
        dis_returns[r]=rewards[r]+gamma*dis_returns[r+1]
    return dis_returns
        
        
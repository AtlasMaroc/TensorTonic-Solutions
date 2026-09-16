def discounted_returns(rewards, gamma):
    """
    Returns: list of G_t values, one per timestep, each rounded to 4 decimals
    """
    discounted_returns = []
    G = 0.0
    
    for r in reversed(rewards): # r in (2,1,0)
        G = r + gamma*G
        discounted_returns.append(G)
    discounted_returns = [round(x,4) for x in discounted_returns]
    # 
    return discounted_returns[::-1]

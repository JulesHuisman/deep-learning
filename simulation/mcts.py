import numpy as np

def mcts(root, search_depth, net, add_noise=False, noise_eps=0, dirichlet_alpha=0):
    """
    Perform one Monte Carlo Tree Search.
    Decides the next play.
    """
    for _ in range(search_depth):
        # print('\nNEW SEARCH ------------------------------------------- \n')
        # Select the best leaf (exploit or explore)
        leaf = root.select_leaf()

        # If the leaf is a winning state
        if leaf.game.won():
            value = (1.18 - (9 * leaf.depth / 350))
            leaf.backprop(-value)
            continue

        # If the leaf is a draw
        if leaf.game.moves() == []:
            leaf.backprop(0)
            continue
        
        # Encode the board for the connect net
        encoded_board = leaf.game.encoded()
        
        # Predict the policy and value of the board state
        policy_estimate, value_estimate = net.predict(encoded_board)
        # policy_estimate, value_estimate = dirichlet_noise(np.array([0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857, 0.142857]), 0.1, 0.2), np.random.normal(0.0, 0.05)

        # Possibly add dirichlet noise to the priors of the root node
        if add_noise and (leaf == root):
            policy_estimate = dirichlet_noise(policy_estimate, noise_eps, dirichlet_alpha)
    
        leaf.expand(policy_estimate)
        leaf.backprop(value_estimate)

    # Return the root with new information
    return root

def dirichlet_noise(priors, noise_eps, dirichlet_alpha):
    """
    Add dirichlet noise to the priors of the root noise to add some randomness to the process
    """
    return (1 - noise_eps) * priors + noise_eps * np.random.dirichlet([dirichlet_alpha] * len(priors))

def get_policy(root, temperature):
    """
    Policy is based on the number of visits to that node.
    Better nodes get visited more often.
    Normalize the number of moves to transform it into a probablity distribution.
    Temperature is a measure of exploration.
    """
    n_visits = root.child_number_visits ** (1 / temperature)
    return n_visits / sum(n_visits)
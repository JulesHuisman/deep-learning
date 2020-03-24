def mcts(root, depth, net, add_noise=False):
    """
    Perform one Monte Carlo Tree Search.
    Decides the next play.
    """
    for _ in range(depth):
        # Select the best leaf (exploit or explore)
        leaf = root.select_leaf()

        # If the leaf is a winning state
        if leaf.game.won():
            leaf.backprop(-1)
            continue

        # If the leaf is a draw
        if leaf.game.moves() == []:
            leaf.backprop(0)
            continue
        
        # Encode the board for the connect net
        encoded_board = leaf.game.encoded()
        
        # Predict the policy and value of the board state
        policy_estimate, value_estimate = net.predict(encoded_board)
    
        leaf.expand(policy_estimate)
        leaf.backprop(value_estimate)

    # Return the root with new information
    return root

def add_noise():
    pass

def get_policy(root, temperature):
    """
    Policy is based on the number of visits to that node.
    Better nodes get visited more often.
    Normalize the number of moves to transform it into a probablity distribution.
    Temperature is a measure of exploration.
    """
    n_visits = root.child_number_visits ** (1 / temperature)
    return n_visits / sum(n_visits)
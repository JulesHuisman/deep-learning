# DeepFour
Project for the deep learning course. The mechanics are inspired by the AlphaZero AI from DeepMind.

You can play against DeepFour here: www.juleshuisman.com

#### Simulations
All parts needed to play games

| Entity  | Location | Description
| --- | --- | --- |
| Game  | `simulation/game.py`  | Representation of a connect four game
| MCTS  | `simulation/mcts.py`  | Performs the monte-carlo tree search
| Nodes  | `simulation/nodes.py`  | Each node represents a game state, needed for the MCTS
| DeepFour  | `deepfour.py`  | The residual neural network

##### game.py

```
simulation/game.py
```
Class used to represent a game. It is able to check if a game was won, and can encode the board state to feed it to the neural network. AlphaZero encodes the current player in a third dimension, however this made DeepFour behave differently depending on the players turn. I adjusted it to only two dimensions, where the first dimension is the (hot) encoded stones of the current player.

##### mcts.py

```
simulation/mcts.py
```
Function to perform the monte-carlo tree search. A value function is added to terminal states to speed up training of the network. Instead of simply using {-1, 1, 0} the value is scaled to how many moves were played in the game (`value = (1.18 - (9 * leaf.depth / 350))`), this makes DeepFour prefer fast wins over slow wins and slow losses over fast losses. 

```
def dirichlet_noise(priors, noise_eps, dirichlet_alpha):
    return (1 - noise_eps) * priors + noise_eps * np.random.dirichlet([dirichlet_alpha] * len(priors))
```
Adding dirichlet noise proved to be import to prevent DeepFour from getting stuck in a feedback loop between MCTS and the neural network. By adding noise to the predicted priors of the neural network you make sure the tree search has some randomness which helps it from avoiding feedback loops.

```
search_depth
```
After a lot of experimentation I found the search depth of the MCTS was a really important parameter. The tree search improves upon the initial policy prediction of the neural network and the neural network predictions are based on the results of the MCTS. If the search depth is too shallow the MCTS cannot improve the neural network predictions which means the whole system deteriorates. I ended up with a search depth of 700.

##### nodes.py
```
simulation/nodes.py
```
These nodes represent game states and perform the MCTS functions (`select`, `expand`, `backprop`). An important aspect is that the value of nodes are being alternated when backpropagating (What is good for green is bad for red).

##### deepfour.py
```
deepfour.py
```
The actual neural network. The structure of the network is as follows:
| Type | Filters / Nodes | Kernel | Activation
| ---  | --- | --- | --- |
| Convolution | 64 | 4x4 | Relu |
| Resblock | 64, 64 | 4x4, 4x4 | Relu |
| Resblock | 64, 64 | 4x4, 4x4 | Relu |
| Resblock | 64, 64 | 4x4, 4x4 | Relu |
| Resblock | 64, 64 | 4x4, 4x4 | Relu |
| Resblock | 64, 64 | 4x4, 4x4 | Relu |
| Policy conv | 2 | 1x1 | Relu |
| Policy dense | 7 |  | Softmax |
| Value conv | 1 | 1x1 | Relu |
| Value dense | 32 |  | Relu |
| Value dense | 1 |  | Tanh |

Overall the network is a (way) more shallow version of the AlphaGo Zero network. But as connect four is simpler game this makes sense. The kernels were changed to 4x4 to capture 16 stones at once.

A L2 regulation value of `0.0001` is introduced to prevent the model from overfitting on the training data.

#### Processes
The three main processes of DeepFour

| Process  | Location | Description
| --- | --- | --- |
| Self play  | `process/play.py`  | Runs the self-play
| Optimize  | `process/optimize.py`  | Trains the neural network
| Evaluate  | `process/evaluate.py`  | The trained neural network battles the previous best network

##### play.py
```
process/play.py
```
This process runs the self-play of DeepFour. This is run in parallel to speed up the game generation. For a MCTS search depth of 700, 3 workers on 16 threads was optimal. However, because of the search depth the game generation was still quite slow. This is because the MCTS needs to request individual board states from the network, which means this process cannot be sped up by using a GPU. AlphaZero uses distributed MCTS which allows batched requests of board states, which drastically speeds up game generation.

Each game is stored on disk for the training process, each board state is also mirrored to enrich the training data. 

For me it took around 8 hours per generation (For a total of 6 generations).

##### optimize.py
```
process/optimize.py
```
This process trains the neural network. There is a memory of all previous moves (for up to 500.000) games. From this memory we take 300 random batches of 512 moves to train the network on. The network is optimized using SGD using a learning rate of 0.001 and a momentum of 0.9. I removed the learning rate schedule as we only reach 6 generations.

##### evaluate.py
```
process/evaluate.py
```
To make sure the new networks are up to par, we let them dual with the previous best network. We play 30 games, if the challenger network wins more than 55% of the games it becomes the new best model.

| Generation | Winrate | Vs | Moves
| ---  | --- | --- | --- |
| 0 (random) | - | - | - |
| 1 | 84% | 0 | 0 - 100.000 |
| 2 | 67% | 1 | 0 - 200.000 |
| 3 | 76% | 2 | 0 - 300.000 |
| 4 | 57% | 3 | 0 - 400.000 |
| 5 | ~~50%~~ | 4 | 0 - 500.000 |
| 6 | 64% | 4 | 100.000 - 600.000 |

##### run.py
```
run.py
```
Entrypoint to run all the different processes. All processes are logged using MLFlow.

`python run.py --process self`
`python run.py --process opt`
`python run.py --process eval`

#### Parameters

All settings can be found in `settings.json`

| Parameter | Value | Description | Reasoning
| ---  | --- | --- | --- |
| Search depth | 700 | How many moves should the MCTS inspect | Deeper search depth improves policy quality
| cPuct | 2 | Should the MCTS focus more on exploration or exploitation | A lower value resulted in missed opportunities. This way the MCTS explores more.
| Exploit turns | 5 | How many moves to play with a high [MCTS policy temperature](https://web.stanford.edu/~surag/posts/alphazero.html) | If you lower this value the MCTS always plays the same kind of moves at the beginning, lowering exploration |
| Dirichlet Alpha | 1.0 | The "aggressiveness" of the dirichlet noise | Increasing this value helped with preventing the system from creating a feedback loop. |
| Dirichlet noise eps | 0.2 | What percentage of dirichlet noise to take | This value strikes a nice balance between using the neural network and using noise. |
| Filters | 64 | How many convolution filters to use | 64 filters created enough capacity, but limited the size of the network to speed up game generation |
| Kernel size | 4 | The size of the convolution filter | A size of 4x4 allows the network to capture 16 stones in which a connection of 4 stones can be made.
| L2 | 0.0001 | L2 regularization | A small L2 value prevents the network from overfitting. |

#### References
https://web.stanford.edu/~surag/posts/alphazero.html

https://github.com/plkmo/AlphaZero_Connect4

https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a

https://medium.com/applied-data-science/how-to-build-your-own-alphazero-ai-using-python-and-keras-7f664945c188

https://medium.com/@jonathan_hui/alphago-zero-a-game-changer-14ef6e45eba5

https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5

https://github.com/Zeta36/connect4-alpha-zero

http://tim.hibal.org/blog/alpha-zero-how-and-why-it-works/

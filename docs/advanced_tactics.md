# Advanced Tic-Tac-Toe Tactics

## Game Theory Classification
Tic-tac-toe is a zero-sum, perfect information game. It was fully solved by computation and is known to always result in a draw with optimal play from both sides. The game tree has 255,168 possible games when considering symmetry.

## Minimax Algorithm
The minimax algorithm is the standard approach for optimal play in tic-tac-toe. It works by recursively evaluating all possible future game states and choosing the move that maximizes the player's minimum guaranteed outcome.

### Alpha-Beta Pruning
Alpha-beta pruning is an optimization of minimax that eliminates branches that cannot influence the final decision. This reduces the number of nodes evaluated from O(b^d) to approximately O(b^(d/2)), where b is the branching factor and d is the depth.

## Position Evaluation

### Square Values by Winning Lines:
- Center (position 4): Part of 4 winning lines — strongest square
- Corners (0, 2, 6, 8): Part of 3 winning lines each — second strongest
- Edges (1, 3, 5, 7): Part of 2 winning lines each — weakest

### Board Symmetry
The board has 8 symmetries (4 rotations × 2 reflections). This means there are only 3 truly distinct opening moves: center, corner, or edge. Exploiting symmetry reduces the effective game tree size.

## First-Mover Advantage
The first player (X) has a slight advantage with 131,184 possible wins compared to 77,904 for the second player (O). However, this advantage disappears with perfect play, resulting in a draw.

## Common Mistakes
1. Not blocking an immediate threat (most common beginner mistake)
2. Playing an edge as the first move (weakest opening)
3. Failing to recognize fork setups
4. Responding to a corner opening with an edge instead of center
5. Not taking the center when available early in the game

## Robot Strategy Notes
When implementing tic-tac-toe for a robot:
- The robot should always play optimally using minimax
- The robot should detect the human's move via computer vision before calculating its response
- Consider adding a small delay before the robot moves to make the game feel more natural
- The robot should handle edge cases like detecting an invalid board state (e.g., two new pieces appearing at once)

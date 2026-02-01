# Tic-Tac-Toe Strategy Guide

## Opening Strategy

The center square (position 4) is the most valuable opening move. It gives access to 4 winning lines: both diagonals, the middle row, and the middle column. A player who takes the center first has the most flexibility.

The corner squares (positions 0, 2, 6, 8) are the second most valuable. Each corner connects to 3 winning lines: one row, one column, and one diagonal. Opening with a corner forces the opponent into a limited set of safe responses.

The edge squares (positions 1, 3, 5, 7) are the weakest opening moves. Each edge only connects to 2 winning lines: one row and one column. Never open with an edge if you can avoid it.

## Defensive Strategy

### Blocking
Always check if your opponent has two in a row before making an offensive move. If they do, you must block immediately or you will lose on the next turn.

### The Fork
A fork is when a player creates two threats simultaneously. The opponent can only block one, so the forking player wins. The most common fork setup is taking two non-adjacent corners with the center, creating two diagonal threats.

### Preventing Forks
If your opponent could create a fork, you need to force them to defend by creating your own two-in-a-row threat, but only if the resulting block does not give them a fork position.

## Common Patterns

### The Corner Trap
If X plays a corner and O plays the center, X should play the opposite corner. This creates a situation where X can fork on the next move regardless of where O plays.

### The Edge Trap
If X plays center and O plays an edge (not a corner), X should play the corner adjacent to O's edge. This leads to a forced win for X.

### Double Corner Setup
Playing two opposite corners (like positions 0 and 8, or 2 and 6) creates a powerful position. If the opponent doesn't play center, this leads to an easy fork.

## Optimal Play Sequences

### X opens center (position 4):
- If O plays corner: X plays opposite corner → X can force a win
- If O plays edge: X plays adjacent corner → X wins with fork
- If O plays center (impossible, taken): N/A

### X opens corner (position 0):
- If O plays center: X plays opposite corner (position 8) → Draw with optimal play
- If O plays edge: X can force a win through forking
- If O plays non-opposite corner: X can force a win

## Win Conditions
There are 8 possible winning lines:
- Three rows: [0,1,2], [3,4,5], [6,7,8]
- Three columns: [0,3,6], [1,4,7], [2,5,8]
- Two diagonals: [0,4,8], [2,4,6]

## Key Principles
1. Always take a win if available
2. Always block an opponent's win
3. Create forks (two ways to win) when possible
4. Block opponent's fork attempts
5. Play center if available
6. Play corners over edges
7. Tic-tac-toe is a solved game - perfect play always results in a draw

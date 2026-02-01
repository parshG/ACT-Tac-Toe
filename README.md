# Tic-Tac-Toe Robot Brain 🤖

CV-based board detection + Minimax AI for your LeRobot SO-101 arm.

## Quick Start

```bash
# Install dependencies
pip install opencv-python numpy

# Run color calibration first (find HSV values for your pieces)
python tictactoe_brain.py --calibrate-colors

# Run the game
python tictactoe_brain.py
```

## Board Position Mapping

```
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
```

## Setup Steps

### 1. Calibrate Colors
Your cubes and shot glasses need distinct colors. Run:
```bash
python tictactoe_brain.py --calibrate-colors
```

Adjust the sliders until only your piece is highlighted white, then press `p` to print the values.

Update these values in `tictactoe_brain.py`:
```python
self.robot_color = ColorRange.from_hsv(
    h_range=(0, 10),      # Your cube color
    s_range=(100, 255),
    v_range=(100, 255)
)
self.human_color = ColorRange.from_hsv(
    h_range=(0, 180),     # Your shot glass color
    s_range=(0, 50),
    v_range=(200, 255)
)
```

### 2. Calibrate Grid
When you run the game, press `c` to calibrate. Click the 4 corners of your board:
1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

### 3. Integration with LeRobot

Edit the `move_arm_to_position()` function in `tictactoe_brain.py` or use the controller class:

```python
from lerobot_integration import TicTacToeController

ttt = TicTacToeController(camera_id=0)
ttt.setup()

while True:
    input("Press Enter after human plays...")
    result = ttt.get_next_move()
    
    if result['game_over']:
        print(f"Winner: {result['winner']}")
        break
    
    position = result['move']
    # Your arm code here:
    # robot.move_to(GRID_POSITIONS[position])
    # robot.place_cube()

ttt.cleanup()
```

## Controls

| Key | Action |
|-----|--------|
| `c` | Calibrate grid |
| `space` | Robot makes move |
| `r` | Reset game |
| `q` | Quit |

## Tips

- **Lighting matters!** Consistent lighting helps color detection
- **Contrasting colors** - use pieces with very different colors
- **Camera angle** - overhead view works best, but perspective transform handles angles
- **Minimum area** - adjust `min_piece_area` if pieces aren't detected (default: 500 pixels)

## Files

- `tictactoe_brain.py` - Main CV detection + Minimax AI
- `lerobot_integration.py` - Examples for integrating with your arm

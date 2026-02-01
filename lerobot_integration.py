"""
LeRobot SO-101 Integration Example
==================================
Shows how to integrate the CV pipeline with your existing arm control code.
"""

from tictactoe_brain import TicTacToeRobot, Player
import time

# =============================================================================
# OPTION 1: Simple Integration (Blocking)
# =============================================================================

def simple_integration():
    """
    Basic usage: Get moves one at a time.
    Good for testing or when you have synchronous arm control.
    """
    robot = TicTacToeRobot(camera_id=0)
    robot.start_camera()
    
    # First: calibrate the grid
    robot.calibrate()
    
    print("\nReady to play! The robot will respond after human moves.")
    print("Press Enter after placing your piece...")
    
    while True:
        input("\n[Press Enter to scan the board and get robot's move]")
        
        move = robot.get_robot_move()
        
        if move is None:
            print("Game over!")
            break
        
        print(f"\n>>> ROBOT SHOULD MOVE TO: Position {move}")
        print(f"    Grid location: Row {move // 3}, Column {move % 3}")
        
        # =====================================================
        # INSERT YOUR LEROBOT ARM CODE HERE
        # =====================================================
        # Example pseudocode:
        # 
        # from lerobot.common.robot_devices.robots.factory import make_robot
        # robot_arm = make_robot("so100")
        # 
        # # Move to pickup position
        # robot_arm.move_to(CUBE_PICKUP_POSITION)
        # robot_arm.gripper_close()
        # 
        # # Move to grid position
        # GRID_POSITIONS = {
        #     0: [x0, y0, z0, ...],
        #     1: [x1, y1, z1, ...],
        #     ...
        # }
        # robot_arm.move_to(GRID_POSITIONS[move])
        # robot_arm.gripper_open()
        # =====================================================
        
        print("\n[Place the robot's cube at the indicated position]")
        input("[Press Enter when done to continue]")
    
    robot.stop_camera()


# =============================================================================
# OPTION 2: Automatic Mode (with state change detection)
# =============================================================================

def automatic_integration():
    """
    Automatic mode: Detects when human has played and responds automatically.
    """
    robot = TicTacToeRobot(camera_id=0)
    robot.start_camera()
    robot.calibrate()
    
    last_human_count = 0
    last_robot_count = 0
    robot_is_thinking = False
    
    print("\nAutomatic mode started!")
    print("The robot will automatically respond when it detects your move.")
    print("Press 'q' to quit.\n")
    
    import cv2
    
    while True:
        frame = robot.capture_frame()
        board = robot.detector.detect_board_state(frame)
        
        human_count = sum(1 for p in board if p == Player.HUMAN)
        robot_count = sum(1 for p in board if p == Player.ROBOT)
        
        # Visualize
        viz = robot.detector.visualize(frame, board)
        cv2.imshow("Tic-Tac-Toe", viz)
        
        # Check if human just played
        if human_count > last_human_count and not robot_is_thinking:
            print(f"Human played! (pieces: Human={human_count}, Robot={robot_count})")
            robot_is_thinking = True
            
            # Give a moment for the board to settle
            time.sleep(0.5)
            
            # Get robot's response
            move = robot.ai.get_best_move(board)
            
            if move is not None:
                print(f">>> ROBOT MOVE: Position {move}")
                
                # ===========================================
                # YOUR ARM CONTROL CODE HERE
                # move_arm_to_position(move)
                # ===========================================
                
                # Wait for the move to be executed
                print("Waiting for arm to complete move...")
                time.sleep(2)  # Adjust based on your arm speed
                
            robot_is_thinking = False
        
        last_human_count = human_count
        last_robot_count = robot_count
        
        # Check for winner
        winner = robot.ai.check_winner(board)
        if winner:
            print(f"\n{'='*40}")
            print(f"GAME OVER: {'Robot' if winner == Player.ROBOT else 'Human'} wins!")
            print(f"{'='*40}")
            break
        
        if robot.ai.is_board_full(board):
            print("\nGAME OVER: It's a draw!")
            break
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    robot.stop_camera()
    cv2.destroyAllWindows()


# =============================================================================
# OPTION 3: Function to call from your existing code
# =============================================================================

class TicTacToeController:
    """
    Controller class you can instantiate from your existing LeRobot code.
    """
    
    def __init__(self, camera_id=0):
        self.robot = TicTacToeRobot(camera_id=camera_id)
        self.calibrated = False
    
    def setup(self):
        """Call this once at the start"""
        self.robot.start_camera()
        self.robot.calibrate()
        self.calibrated = True
    
    def get_next_move(self) -> dict:
        """
        Call this to get the robot's next move.
        
        Returns:
            dict with keys:
                - 'move': int (0-8) or None if game over
                - 'row': int (0-2)
                - 'col': int (0-2)
                - 'board': current board state
                - 'game_over': bool
                - 'winner': 'robot', 'human', 'draw', or None
        """
        if not self.calibrated:
            raise RuntimeError("Call setup() first!")
        
        frame = self.robot.capture_frame()
        board = self.robot.detector.detect_board_state(frame)
        
        winner = self.robot.ai.check_winner(board)
        is_full = self.robot.ai.is_board_full(board)
        
        if winner:
            winner_str = 'robot' if winner == Player.ROBOT else 'human'
        elif is_full:
            winner_str = 'draw'
        else:
            winner_str = None
        
        move = self.robot.ai.get_best_move(board)
        
        return {
            'move': move,
            'row': move // 3 if move is not None else None,
            'col': move % 3 if move is not None else None,
            'board': [(p.name, p.value) for p in board],
            'game_over': winner is not None or is_full,
            'winner': winner_str
        }
    
    def cleanup(self):
        """Call when done"""
        self.robot.stop_camera()


# Example usage of the controller class:
"""
from lerobot_integration import TicTacToeController

# In your main LeRobot script:
ttt = TicTacToeController(camera_id=0)
ttt.setup()

while True:
    input("Press Enter after human plays...")
    
    result = ttt.get_next_move()
    
    if result['game_over']:
        print(f"Game over! Winner: {result['winner']}")
        break
    
    position = result['move']
    print(f"Moving arm to position {position}")
    
    # Your arm control code:
    # robot_arm.move_to_grid_position(position)
    # robot_arm.place_cube()

ttt.cleanup()
"""


if __name__ == "__main__":
    print("Select mode:")
    print("1. Simple (manual turn-by-turn)")
    print("2. Automatic (auto-detects human moves)")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "2":
        automatic_integration()
    else:
        simple_integration()

"""
Tic-Tac-Toe Robot Brain
=======================
CV-based board detection + Minimax game logic for LeRobot SO-101 arm

Pipeline: Camera → Detect Board → Game Logic → Output Position (0-8)

Board positions:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class Player(Enum):
    EMPTY = 0
    ROBOT = 1   # Cubes
    HUMAN = 2   # Shot glasses


@dataclass
class ColorRange:
    """HSV color range for detection"""
    lower: np.ndarray
    upper: np.ndarray
    
    @classmethod
    def from_hsv(cls, h_range: Tuple[int, int], s_range: Tuple[int, int], v_range: Tuple[int, int]):
        return cls(
            lower=np.array([h_range[0], s_range[0], v_range[0]]),
            upper=np.array([h_range[1], s_range[1], v_range[1]])
        )


class BoardDetector:
    """
    Detects tic-tac-toe board state using color-based piece detection.
    
    Calibrate the HSV ranges for your specific cubes and shot glasses!
    """
    
    def __init__(
        self,
        # Grid corners in pixel coordinates (top-left, top-right, bottom-right, bottom-left)
        grid_corners: Optional[List[Tuple[int, int]]] = None,
        # HSV ranges - CALIBRATE THESE FOR YOUR SETUP
        robot_color: ColorRange = None,  # Cube color
        human_color: ColorRange = None,  # Shot glass color
        min_piece_area: int = 500,  # Minimum contour area to count as a piece
    ):
        self.grid_corners = grid_corners
        self.min_piece_area = min_piece_area
        
        # Default colors - RED cubes, WHITE/CLEAR shot glasses
        # YOU SHOULD CALIBRATE THESE!
        self.robot_color = robot_color or ColorRange.from_hsv(
            h_range=(0, 10),      # Red hue (also check 170-180 for red wraparound)
            s_range=(100, 255),   # High saturation
            v_range=(100, 255)    # Bright
        )
        self.human_color = human_color or ColorRange.from_hsv(
            h_range=(0, 180),     # Any hue (white is low saturation)
            s_range=(0, 50),      # Low saturation = white/clear
            v_range=(200, 255)    # Very bright
        )
        
        # Red has two ranges in HSV (wraps around)
        self.robot_color_alt = ColorRange.from_hsv(
            h_range=(170, 180),
            s_range=(100, 255),
            v_range=(100, 255)
        )
    
    def calibrate_grid(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Interactive grid calibration - click the 4 corners of your board.
        Returns corners in order: top-left, top-right, bottom-right, bottom-left
        """
        corners = []
        window_name = "Click 4 corners: TL, TR, BR, BL (press 'r' to reset, 'q' when done)"
        
        def click_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
                corners.append((x, y))
                print(f"Corner {len(corners)}: ({x}, {y})")
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, click_callback)
        
        while True:
            display = frame.copy()
            
            # Draw existing corners
            for i, corner in enumerate(corners):
                cv2.circle(display, corner, 8, (0, 255, 0), -1)
                cv2.putText(display, str(i+1), (corner[0]+10, corner[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw grid preview if all corners selected
            if len(corners) == 4:
                pts = np.array(corners, dtype=np.int32)
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                cv2.putText(display, "Press 'q' to confirm, 'r' to reset",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):
                corners.clear()
            elif key == ord('q') and len(corners) == 4:
                break
        
        cv2.destroyWindow(window_name)
        self.grid_corners = corners
        return corners
    
    def get_cell_centers(self) -> List[Tuple[int, int]]:
        """Get pixel coordinates of all 9 cell centers"""
        if not self.grid_corners:
            raise ValueError("Grid not calibrated! Call calibrate_grid() first.")
        
        # Perspective transform to get normalized grid
        src_pts = np.array(self.grid_corners, dtype=np.float32)
        dst_pts = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype=np.float32)
        
        # Inverse transform to map cell centers back to image coordinates
        M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        
        centers = []
        for row in range(3):
            for col in range(3):
                # Center of each cell in normalized coordinates
                cx = col * 100 + 50
                cy = row * 100 + 50
                
                # Transform back to image coordinates
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(pt, M_inv)
                centers.append((int(transformed[0][0][0]), int(transformed[0][0][1])))
        
        return centers
    
    def get_cell_regions(self, frame: np.ndarray) -> List[np.ndarray]:
        """Extract 9 cell regions from the frame"""
        if not self.grid_corners:
            raise ValueError("Grid not calibrated!")
        
        # Warp to top-down view
        src_pts = np.array(self.grid_corners, dtype=np.float32)
        dst_pts = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(frame, M, (300, 300))
        
        # Extract each cell (100x100 pixels each)
        cells = []
        for row in range(3):
            for col in range(3):
                cell = warped[row*100:(row+1)*100, col*100:(col+1)*100]
                cells.append(cell)
        
        return cells
    
    def detect_piece_in_cell(self, cell: np.ndarray) -> Player:
        """Detect what piece (if any) is in a cell using color detection"""
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        
        # Check for robot piece (cube)
        mask_robot1 = cv2.inRange(hsv, self.robot_color.lower, self.robot_color.upper)
        mask_robot2 = cv2.inRange(hsv, self.robot_color_alt.lower, self.robot_color_alt.upper)
        mask_robot = cv2.bitwise_or(mask_robot1, mask_robot2)
        
        # Check for human piece (shot glass)
        mask_human = cv2.inRange(hsv, self.human_color.lower, self.human_color.upper)
        
        # Clean up masks
        kernel = np.ones((5, 5), np.uint8)
        mask_robot = cv2.morphologyEx(mask_robot, cv2.MORPH_OPEN, kernel)
        mask_human = cv2.morphologyEx(mask_human, cv2.MORPH_OPEN, kernel)
        
        robot_area = cv2.countNonZero(mask_robot)
        human_area = cv2.countNonZero(mask_human)
        
        if robot_area > self.min_piece_area:
            return Player.ROBOT
        elif human_area > self.min_piece_area:
            return Player.HUMAN
        else:
            return Player.EMPTY
    
    def detect_board_state(self, frame: np.ndarray) -> List[Player]:
        """
        Detect the full board state from a camera frame.
        Returns list of 9 Player values (positions 0-8)
        """
        cells = self.get_cell_regions(frame)
        return [self.detect_piece_in_cell(cell) for cell in cells]
    
    def visualize(self, frame: np.ndarray, board_state: List[Player]) -> np.ndarray:
        """Draw the detected board state on the frame for debugging"""
        display = frame.copy()
        
        if self.grid_corners:
            # Draw grid
            pts = np.array(self.grid_corners, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            
            # Draw cell contents
            centers = self.get_cell_centers()
            for i, (center, player) in enumerate(zip(centers, board_state)):
                if player == Player.ROBOT:
                    cv2.circle(display, center, 25, (0, 0, 255), 3)  # Red circle
                    cv2.putText(display, "R", (center[0]-10, center[1]+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif player == Player.HUMAN:
                    cv2.drawMarker(display, center, (255, 0, 0), cv2.MARKER_CROSS, 40, 3)
                    cv2.putText(display, "H", (center[0]-10, center[1]+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                else:
                    cv2.putText(display, str(i), (center[0]-10, center[1]+10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        return display


class TicTacToeAI:
    """
    Minimax-based AI for optimal tic-tac-toe play.
    The robot NEVER loses (best case: win, worst case: draw)
    """
    
    WIN_LINES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    
    def __init__(self, robot_player: Player = Player.ROBOT):
        self.robot = robot_player
        self.human = Player.HUMAN if robot_player == Player.ROBOT else Player.ROBOT
    
    def check_winner(self, board: List[Player]) -> Optional[Player]:
        """Check if there's a winner. Returns winner or None."""
        for line in self.WIN_LINES:
            if board[line[0]] != Player.EMPTY and \
               board[line[0]] == board[line[1]] == board[line[2]]:
                return board[line[0]]
        return None
    
    def is_board_full(self, board: List[Player]) -> bool:
        return all(cell != Player.EMPTY for cell in board)
    
    def get_empty_cells(self, board: List[Player]) -> List[int]:
        return [i for i, cell in enumerate(board) if cell == Player.EMPTY]
    
    def minimax(self, board: List[Player], is_maximizing: bool, alpha: float = float('-inf'), beta: float = float('inf')) -> Tuple[int, Optional[int]]:
        """
        Minimax with alpha-beta pruning.
        Returns (score, best_move)
        """
        winner = self.check_winner(board)
        
        # Terminal states
        if winner == self.robot:
            return (10, None)
        elif winner == self.human:
            return (-10, None)
        elif self.is_board_full(board):
            return (0, None)
        
        empty_cells = self.get_empty_cells(board)
        best_move = empty_cells[0]  # Default to first available
        
        if is_maximizing:
            max_eval = float('-inf')
            for cell in empty_cells:
                board[cell] = self.robot
                eval_score, _ = self.minimax(board, False, alpha, beta)
                board[cell] = Player.EMPTY
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = cell
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return (max_eval, best_move)
        else:
            min_eval = float('inf')
            for cell in empty_cells:
                board[cell] = self.human
                eval_score, _ = self.minimax(board, True, alpha, beta)
                board[cell] = Player.EMPTY
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = cell
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return (min_eval, best_move)
    
    def get_best_move(self, board: List[Player]) -> Optional[int]:
        """
        Get the best move for the robot.
        Returns position 0-8, or None if no moves available.
        """
        if self.is_board_full(board) or self.check_winner(board):
            return None
        
        _, best_move = self.minimax(board.copy(), True)
        return best_move


class TicTacToeRobot:
    """
    Main controller that combines CV detection with game logic.
    """
    
    def __init__(self, camera_id: int = 0):
        self.detector = BoardDetector()
        self.ai = TicTacToeAI()
        self.camera_id = camera_id
        self.cap = None
        self.last_board_state = [Player.EMPTY] * 9
    
    def start_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        # Give camera time to adjust
        for _ in range(10):
            self.cap.read()
    
    def stop_camera(self):
        """Release camera"""
        if self.cap:
            self.cap.release()
    
    def capture_frame(self) -> np.ndarray:
        """Capture a single frame"""
        if not self.cap:
            self.start_camera()
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame
    
    def calibrate(self):
        """Run interactive calibration"""
        frame = self.capture_frame()
        print("Click the 4 corners of your tic-tac-toe grid:")
        print("  1. Top-left")
        print("  2. Top-right") 
        print("  3. Bottom-right")
        print("  4. Bottom-left")
        self.detector.calibrate_grid(frame)
        print("Calibration complete!")
    
    def get_robot_move(self) -> Optional[int]:
        """
        Main function: Capture frame, detect board, compute best move.
        Returns position 0-8 for the arm to move to, or None if game over.
        """
        frame = self.capture_frame()
        board_state = self.detector.detect_board_state(frame)
        self.last_board_state = board_state
        
        # Debug visualization
        viz = self.detector.visualize(frame, board_state)
        cv2.imshow("Board Detection", viz)
        cv2.waitKey(1)
        
        return self.ai.get_best_move(board_state)
    
    def run_game_loop(self, move_callback=None):
        """
        Run continuous game loop.
        
        Args:
            move_callback: Function to call with the move position (0-8)
                          This is where you integrate with your LeRobot arm!
        """
        print("\n" + "="*50)
        print("TIC-TAC-TOE ROBOT")
        print("="*50)
        print("Controls:")
        print("  'c' - Calibrate grid")
        print("  'space' - Robot makes a move")
        print("  'r' - Reset game")
        print("  'q' - Quit")
        print("="*50 + "\n")
        
        self.start_camera()
        
        try:
            while True:
                frame = self.capture_frame()
                
                # Always try to detect and visualize
                try:
                    board_state = self.detector.detect_board_state(frame)
                    self.last_board_state = board_state
                    viz = self.detector.visualize(frame, board_state)
                except ValueError:
                    viz = frame.copy()
                    cv2.putText(viz, "Press 'c' to calibrate grid", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Check game status
                winner = self.ai.check_winner(self.last_board_state)
                if winner:
                    status = "ROBOT WINS!" if winner == Player.ROBOT else "HUMAN WINS!"
                    cv2.putText(viz, status, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                elif self.ai.is_board_full(self.last_board_state):
                    cv2.putText(viz, "DRAW!", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                cv2.imshow("Tic-Tac-Toe Robot", viz)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.detector.calibrate_grid(frame)
                elif key == ord('r'):
                    self.last_board_state = [Player.EMPTY] * 9
                    print("Game reset!")
                elif key == ord(' '):
                    move = self.ai.get_best_move(self.last_board_state)
                    if move is not None:
                        print(f"\n>>> ROBOT MOVE: Position {move}")
                        print(f"    (Row {move // 3}, Col {move % 3})")
                        
                        if move_callback:
                            move_callback(move)
                    else:
                        print("No valid moves - game over!")
        
        finally:
            self.stop_camera()
            cv2.destroyAllWindows()


# =============================================================================
# COLOR CALIBRATION UTILITY
# =============================================================================

def calibrate_colors(camera_id: int = 0):
    """
    Interactive HSV color calibration tool.
    Use this to find the right color ranges for your pieces!
    """
    cap = cv2.VideoCapture(camera_id)
    
    cv2.namedWindow("Color Calibration")
    cv2.createTrackbar("H Min", "Color Calibration", 0, 180, lambda x: None)
    cv2.createTrackbar("H Max", "Color Calibration", 180, 180, lambda x: None)
    cv2.createTrackbar("S Min", "Color Calibration", 0, 255, lambda x: None)
    cv2.createTrackbar("S Max", "Color Calibration", 255, 255, lambda x: None)
    cv2.createTrackbar("V Min", "Color Calibration", 0, 255, lambda x: None)
    cv2.createTrackbar("V Max", "Color Calibration", 255, 255, lambda x: None)
    
    print("Adjust trackbars until only your target piece is highlighted in white.")
    print("Press 'p' to print current values, 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        h_min = cv2.getTrackbarPos("H Min", "Color Calibration")
        h_max = cv2.getTrackbarPos("H Max", "Color Calibration")
        s_min = cv2.getTrackbarPos("S Min", "Color Calibration")
        s_max = cv2.getTrackbarPos("S Max", "Color Calibration")
        v_min = cv2.getTrackbarPos("V Min", "Color Calibration")
        v_max = cv2.getTrackbarPos("V Max", "Color Calibration")
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        display = np.hstack([frame, result])
        cv2.imshow("Color Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print(f"\nColorRange.from_hsv(")
            print(f"    h_range=({h_min}, {h_max}),")
            print(f"    s_range=({s_min}, {s_max}),")
            print(f"    v_range=({v_min}, {v_max})")
            print(f")")
    
    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--calibrate-colors":
        # Run color calibration
        calibrate_colors()
    else:
        # Example: Define your arm movement function
        def move_arm_to_position(position: int):
            """
            Replace this with your LeRobot SO-101 arm control code!
            
            Position mapping (0-8):
                0 | 1 | 2
                ---------
                3 | 4 | 5
                ---------
                6 | 7 | 8
            """
            print(f"[ARM] Moving to position {position}...")
            # Your LeRobot code here, e.g.:
            # robot.move_to_grid_position(position)
            # robot.place_cube()
        
        # Run the game
        robot = TicTacToeRobot(camera_id=0)
        robot.run_game_loop(move_callback=move_arm_to_position)

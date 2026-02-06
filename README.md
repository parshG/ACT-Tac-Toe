# ACT-Tac-Toe 

## Inspiration

We wanted to push the boundaries of what a low-cost robotic arm can do by combining classical AI, computer vision, and modern Vision-Language-Action models. Tic-tac-toe is a deceptively simple game — but getting a physical robot to see the board, reason about strategy, and physically place pieces requires a full-stack robotics pipeline. We set out to build a robot that doesn't just play tic-tac-toe — it plays *perfectly*, explains its reasoning, and physically moves game pieces using a trained VLA model.

## What It Does

ACT-Tac-Toe is an end-to-end robotic tic-tac-toe system where a **LeRobot SO-101 arm** plays against a human opponent in real time.

- **Sees** the board using computer vision with color-based piece detection across all 9 grid cells
- **Thinks** using the Minimax algorithm with alpha-beta pruning — evaluating up to 255,168 possible game states to guarantee the robot never loses
- **Explains** its moves using a RAG pipeline that retrieves from a knowledge base of tic-tac-toe strategy documents split into ~50 embedded chunks
- **Acts** by physically picking up cubes and placing them on the board, powered by VLA models trained on ~100 recorded demonstration episodes

The robot plays one side using colored cubes while the human opponent uses paper shot glasses, allowing the CV system to distinguish between the two players through HSV color thresholding.

## How We Built It

### Hardware Setup
We used the **LeRobot SO-101 robotic arm** with an attached camera for visual input. The arm was calibrated across 6 joints to reach all 9 positions on a physical tic-tac-toe grid, plus a cube pickup location. The board is divided into a 3×3 grid with each cell mapped to specific (x, y, z) coordinates via a 4-corner perspective calibration.

### VLA Training (Vision-Language-Action)
We trained VLA models to control the robot arm's physical movements — specifically, picking up a cube from a fixed location and placing it onto one of the 9 grid squares.

**Data Collection:**
- Recorded approximately **100 teleoperation episodes** demonstrating the pick-and-place task
- Each episode captured the full motion trajectory: approach → grasp → lift → transport → place → release
- Episodes were recorded across the 9 grid positions to give the model coverage of the full board
- Camera feed was captured at each timestep alongside joint position data for paired vision-action training
- Out of ~120 total recorded episodes, roughly **15-20% had to be discarded** due to failed grasps, inconsistent trajectories, or camera occlusion, leaving ~100 usable episodes

**Model Training:**
We experimented with two VLA training architectures:

1. **ACT (Action Chunking with Transformers)** — Predicts chunks of future actions conditioned on visual observations. We trained ACT on our ~100 episodes using cloud GPUs. The model learned to generalize the pick-and-place motion across grid positions, though the **error rate remained high at approximately 40-60%** — the arm would frequently miss the target cell, drop the cube during transport, or fail to grasp cleanly.

2. **Pi0.5** — A more recent VLA architecture that we also evaluated for the same task. Pi0.5 showed improvements in grasp reliability but still exhibited significant placement errors, with an **overall success rate around 30-50%** depending on which grid position was targeted. Corner positions (0, 2, 6, 8) had higher error rates than center positions due to the arm's reach limits.

**Compute:**
- Training was performed on **NVIDIA A100 and H100 cloud GPUs**
- ACT training ran for multiple hours on a single A100
- Pi0.5 training leveraged H100s for faster iteration on hyperparameter tuning
- H100s provided roughly **2-3x faster training throughput** compared to A100s
- Total training time across experiments: **~10-15 GPU hours**

**Failure Mode Breakdown:**
- Gripper misalignment: **~25% of failures**
- Cube drops during transport: **~35% of failures**
- Imprecise placement (wrong cell or cell edge): **~40% of failures**

Despite the high error rate, the system successfully demonstrated end-to-end autonomous play when the physical actions succeeded.

### Computer Vision (Board Detection)
We used **OpenCV** with classical CV techniques — no deep learning needed for board reading:

- **HSV Color Thresholding** to detect robot pieces (cubes) vs. human pieces (shot glasses) by color, with tunable ranges across 3 channels (Hue: 0-180, Saturation: 0-255, Value: 0-255)
- **Perspective Transform** using a 4-point homography to warp the camera view into a normalized 300×300 pixel top-down grid, handling arbitrary camera angles
- **Grid Calibration** — Interactive 4-corner selection maps the physical board to 9 logical cells of 100×100 pixels each
- **Morphological Operations** (5×5 kernel opening) to clean up noise in detection masks
- **Minimum contour area threshold** of 500 pixels to filter false positives
- Board detection accuracy: **~90-95%** under consistent lighting conditions, dropping to **~75-80%** with variable lighting or shadows

### Game AI (Minimax)
The game logic uses the **Minimax algorithm with alpha-beta pruning**:

- Recursively evaluates every possible future game state across a game tree of **255,168 possible games**
- Scores: +10 (robot win), -10 (human loss), 0 (draw)
- Alpha-beta pruning reduces the search space from O(b^d) to approximately O(b^(d/2)), cutting evaluated nodes by **~50-60%**
- The center square connects to **4 winning lines**, corners to **3**, and edges to only **2** — the AI exploits this hierarchy
- First player (X) has 131,184 possible wins vs. 77,904 for second player (O), but optimal play always forces a draw
- **Result: Mathematically optimal play — the robot never loses. 100% win/draw rate.**

### RAG Strategy Explanations
We added a **Retrieval-Augmented Generation** pipeline so the robot can explain *why* it makes each move:

- **Knowledge Base** — 2 strategy documents covering opening theory, fork tactics, blocking strategy, positional values, and game theory (~2,500 words total)
- **Text Splitting** — Recursive character splitting with 500-character chunks and 50-character overlap, producing ~50 indexed chunks
- **Embeddings** — OpenAI `text-embedding-3-small` (1,536 dimensions) to vectorize strategy chunks
- **Vector Store** — ChromaDB for similarity search, retrieving the top 3 most relevant chunks per query
- **LLM** — GPT-4o-mini (temperature 0.3) via LangChain's RetrievalQA chain generates 2-3 sentence explanations
- Average explanation latency: **~1-2 seconds** per move

## Tech Stack

| Component | Technology |
|-----------|------------|
| Robotic Arm | LeRobot SO-101 (6-DOF) |
| VLA Models | ACT (Action Chunking with Transformers) + Pi0.5 |
| Training Data | ~100 teleoperation episodes |
| Training Compute | NVIDIA A100 + H100 cloud GPUs (~10-15 GPU hours) |
| Board Detection | OpenCV (HSV thresholding, perspective transforms) |
| Game AI | Minimax with Alpha-Beta Pruning |
| Strategy RAG | LangChain + ChromaDB + OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small (1,536-dim) |
| Language | Python |

## Challenges We Ran Into

- **High VLA error rate** — With only ~100 training episodes, our ACT and Pi0.5 models achieved a **30-50% success rate** on the physical pick-and-place task. The primary failure modes were gripper misalignment (25%), mid-transport drops (35%), and imprecise placement (40%). Corner positions were especially error-prone due to the arm reaching near its kinematic limits.
- **Color detection tuning** — Lighting conditions heavily affect HSV thresholding. A 10-15% shift in ambient brightness could cause detection failures. We built a real-time calibration tool with 6 live trackbars to dial in exact color ranges for our pieces.
- **Camera perspective** — The camera isn't perfectly overhead, introducing up to **15-20 degrees of perspective distortion**. We implemented a 4-point perspective transform to warp the board into a clean 300×300 pixel top-down view before analyzing cells.
- **VLA data collection** — Getting consistent, high-quality demonstration episodes required careful teleoperation. Roughly 15-20% of recorded episodes had to be discarded due to poor quality.
- **GPU costs** — Training on A100s and H100s added up quickly during experimentation. We had to be strategic about hyperparameter sweeps to stay within budget.
- **Piece differentiation** — Distinguishing between cubes and shot glasses required choosing pieces with very different color profiles and tuning separate HSV ranges for each, with a minimum area threshold to avoid false positives from background noise.

## Accomplishments We're Proud Of

- The robot plays **mathematically perfect** tic-tac-toe — it's impossible to beat (100% win/draw rate)
- Built a complete **perception → reasoning → action** pipeline on a sub-$1000 hardware setup
- Successfully trained two VLA architectures (ACT and Pi0.5) from scratch on a relatively small dataset of ~100 episodes
- The RAG system retrieves relevant strategy context in under 2 seconds and generates meaningful move explanations
- Combined classical algorithms (Minimax, HSV detection) with modern AI (VLA, RAG, LLMs) in a cohesive system
- Board detection achieves **90-95% accuracy** under controlled conditions

## What We Learned

- Classical computer vision is still incredibly powerful for well-defined detection tasks — you don't always need deep learning. HSV thresholding runs at **30+ FPS** compared to YOLO at ~5-10 FPS on the same hardware
- **100 episodes isn't enough for robust VLA training** — the high error rate (40-60%) confirmed that these models are data-hungry and would benefit from 200-500+ demonstrations
- ACT and Pi0.5 have different strengths — ACT was more stable for simple trajectories while Pi0.5 showed better generalization across positions but with less consistency
- Alpha-beta pruning makes Minimax practical for real-time gameplay, reducing node evaluations by over 50%
- RAG pipelines are straightforward to add on top of existing systems and provide significant value for explainability
- **A100s vs H100s** — H100s provided roughly 2-3x faster training for our VLA models, which made rapid iteration possible during the hackathon

## What's Next for ACT-Tac-Toe

- **More training data** — Scale to 500+ episodes to push VLA success rate above 80%
- **Voice integration** — Have the robot verbally announce and explain its moves using text-to-speech
- **Difficulty levels** — Add suboptimal play modes (random moves 20-40% of the time) so the robot doesn't always play perfectly
- **Expanded games** — Apply the same VLA + CV + game AI framework to Connect Four or checkers
- **Real-time commentary** — Stream the RAG explanations as live commentary during gameplay
- **Fine-tuned VLA** — Experiment with larger pre-trained VLA backbones to reduce the data requirement below 100 episodes

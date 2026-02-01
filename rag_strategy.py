"""
Tic-Tac-Toe Strategy RAG Module
================================
Uses LangChain + OpenAI + ChromaDB to retrieve tic-tac-toe strategy
knowledge and explain the robot's moves.

Setup:
    pip install langchain langchain-openai langchain-community chromadb
    export OPENAI_API_KEY="your-key-here"
"""

import os
from pathlib import Path
from typing import Optional, List

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Adjust this to wherever your strategy docs live
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


class StrategyRAG:
    """
    RAG pipeline that indexes tic-tac-toe strategy documents
    and answers questions about moves and strategy.
    """

    def __init__(self, docs_dir: str = DOCS_DIR, persist_dir: str = CHROMA_DIR):
        self.docs_dir = docs_dir
        self.persist_dir = persist_dir

        # Models
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        # Vector store (loaded or built on first use)
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain = None

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_docs(self, force_rebuild: bool = False):
        """Load strategy docs from disk, split, embed, and store."""
        # Reuse existing index if available
        if not force_rebuild and os.path.exists(self.persist_dir):
            print("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            self._build_chain()
            return

        print(f"Indexing documents from {self.docs_dir} ...")
        loader = DirectoryLoader(
            self.docs_dir,
            glob="**/*.md",
            loader_cls=TextLoader,
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s)")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
        )
        chunks = splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        print("Vector store built and persisted.")
        self._build_chain()

    # ------------------------------------------------------------------
    # Chain setup
    # ------------------------------------------------------------------

    PROMPT_TEMPLATE = """\
You are a tic-tac-toe strategy expert commentating on a robot vs human game.
Use the retrieved strategy context to explain moves clearly and concisely.

Board positions are numbered:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8

Context:
{context}

Question: {question}

Give a short, insightful explanation (2-3 sentences max). Be specific about
why the move is strategically good or bad.
"""

    def _build_chain(self):
        if self.vectorstore is None:
            raise RuntimeError("Call index_docs() first")

        prompt = PromptTemplate(
            template=self.PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3},
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def explain_move(self, board_state: list, move: int, is_robot: bool = True) -> str:
        """
        Explain why a move is being made given the current board state.

        Args:
            board_state: list of 9 values (0=empty, 1=robot, 2=human)
            move: position 0-8 the robot chose
            is_robot: True if this is the robot's move
        """
        if self.qa_chain is None:
            self.index_docs()

        player = "Robot" if is_robot else "Human"

        # Build a readable board
        symbols = {0: ".", 1: "R", 2: "H"}
        board_str = ""
        for row in range(3):
            cells = [symbols.get(board_state[row * 3 + col], "?") for col in range(3)]
            board_str += " | ".join(cells) + "\n"

        # Count moves to determine game phase
        total_moves = sum(1 for c in board_state if c != 0)
        phase = "opening" if total_moves < 2 else "midgame" if total_moves < 5 else "endgame"

        question = (
            f"The current board is:\n{board_str}\n"
            f"It's the {phase}. {player} is about to play position {move} "
            f"(row {move // 3}, col {move % 3}). "
            f"Why is this a good move? What strategy does it follow?"
        )

        result = self.qa_chain.invoke({"query": question})
        return result["result"]

    def query(self, question: str) -> str:
        """Ask any strategy question."""
        if self.qa_chain is None:
            self.index_docs()
        result = self.qa_chain.invoke({"query": question})
        return result["result"]

    def get_relevant_strategy(self, question: str) -> List[str]:
        """Return raw retrieved chunks (useful for debugging)."""
        if self.vectorstore is None:
            self.index_docs()
        docs = self.vectorstore.similarity_search(question, k=3)
        return [doc.page_content for doc in docs]


# ======================================================================
# Integration with TicTacToeRobot
# ======================================================================

class SmartTicTacToeRobot:
    """
    Extends the base robot with RAG-powered move explanations.
    """

    def __init__(self, camera_id: int = 0):
        from tictactoe_brain import TicTacToeRobot, Player
        self.robot = TicTacToeRobot(camera_id=camera_id)
        self.rag = StrategyRAG()
        self.rag.index_docs()
        self.Player = Player

    def get_move_with_explanation(self) -> dict:
        """
        Get the robot's next move along with a strategic explanation.

        Returns:
            {
                'move': int (0-8),
                'explanation': str,
                'board': list,
                'game_over': bool,
                'winner': str or None
            }
        """
        frame = self.robot.capture_frame()
        board_state = self.robot.detector.detect_board_state(frame)

        winner = self.robot.ai.check_winner(board_state)
        is_full = self.robot.ai.is_board_full(board_state)

        if winner:
            winner_str = "robot" if winner == self.Player.ROBOT else "human"
        elif is_full:
            winner_str = "draw"
        else:
            winner_str = None

        move = self.robot.ai.get_best_move(board_state)
        explanation = None

        if move is not None:
            board_ints = [p.value for p in board_state]
            explanation = self.rag.explain_move(board_ints, move, is_robot=True)

        return {
            "move": move,
            "explanation": explanation,
            "board": [(p.name, p.value) for p in board_state],
            "game_over": winner is not None or is_full,
            "winner": winner_str,
        }


# ======================================================================
# CLI demo (works without camera)
# ======================================================================

def demo_no_camera():
    """
    Demo the RAG pipeline with a simulated board — no camera needed.
    """
    print("Initializing RAG pipeline...")
    rag = StrategyRAG()
    rag.index_docs()

    print("\n" + "=" * 50)
    print("TIC-TAC-TOE STRATEGY RAG DEMO")
    print("=" * 50)

    # --- Example 1: Explain an opening move ---
    print("\n--- Example 1: Opening move ---")
    board = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # empty board
    move = 4  # center
    explanation = rag.explain_move(board, move, is_robot=True)
    print(f"Board: empty")
    print(f"Robot plays: position {move} (center)")
    print(f"Explanation: {explanation}")

    # --- Example 2: Blocking move ---
    print("\n--- Example 2: Blocking move ---")
    board = [1, 2, 0, 0, 2, 0, 0, 0, 0]  # human has 1,4 → threatening 7
    move = 7
    explanation = rag.explain_move(board, move, is_robot=True)
    print(f"Board: R . .  /  . H .  /  . . .")
    print(f"Human threatening column! Robot plays: position {move}")
    print(f"Explanation: {explanation}")

    # --- Example 3: Free-form question ---
    print("\n--- Example 3: Strategy question ---")
    question = "What is a fork in tic-tac-toe and how do I create one?"
    answer = rag.query(question)
    print(f"Q: {question}")
    print(f"A: {answer}")

    # --- Interactive mode ---
    print("\n--- Interactive mode (type 'quit' to exit) ---")
    while True:
        q = input("\nAsk a strategy question: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        answer = rag.query(q)
        print(f"\n{answer}")


if __name__ == "__main__":
    demo_no_camera()

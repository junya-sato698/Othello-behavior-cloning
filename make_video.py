import numpy as np
import torch
import torch.nn as nn
import pygame
import sys
import time
from tqdm import tqdm
import cv2

# --- 設定 ---
MODEL_PATH = "bc_model_deep_cnn_best.pth"
PATTERN = 1
OUTPUT_VIDEO_FILE = "othello_presentation_multi.mp4" 

# 表示設定
WINDOW_SIZE = 600
GRID_SIZE = 8
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
MOVE_INTERVAL = 1.0

# 動画設定
VIDEO_FPS = 30 
FRAMES_PER_MOVE = int(VIDEO_FPS * MOVE_INTERVAL) 

# 色定義
COLOR_BG = (34, 139, 34)
COLOR_LINE = (0, 0, 0)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_VALID = (255, 0, 0)
COLOR_GOLD = (255, 215, 0)
COLOR_LAST = (0, 255, 255)

# --- ゲームロジック ---
class OthelloEnv:
    # オセロのゲームルールと盤面状態を管理するクラス
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[3, 3] = 1; self.board[4, 4] = 1
        self.board[3, 4] = -1; self.board[4, 3] = -1
        self.current_player = 1
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def get_valid_moves(self, player):
        # 指定されたプレイヤーの有効手を取得する
        valid = []
        for r in range(8):
            for c in range(8):
                if self.board[r, c] == 0 and self._check_flips(r, c, player):
                    valid.append((r, c))
        return valid

    def _check_flips(self, r, c, player):
        # 裏返る石があるか判定する
        opponent = -player
        for dr, dc in self.directions:
            cr, cc = r + dr, c + dc
            has_opp = False
            while 0 <= cr < 8 and 0 <= cc < 8:
                if self.board[cr, cc] == opponent: has_opp = True
                elif self.board[cr, cc] == player:
                    if has_opp: return True
                    break
                else: break
                cr, cc = cr + dr, cc + dc
        return False

    def make_move(self, move, player):
        # 石を置き、挟んだ石を裏返す
        r, c = move
        self.board[r, c] = player
        opponent = -player
        flips = []
        for dr, dc in self.directions:
            cr, cc = r + dr, c + dc
            potential = []
            while 0 <= cr < 8 and 0 <= cc < 8:
                if self.board[cr, cc] == opponent: potential.append((cr, cc))
                elif self.board[cr, cc] == player:
                    flips.extend(potential); break
                else: break
                cr, cc = cr + dr, cc + dc
        for fr, fc in flips: self.board[fr, fc] = player
        return flips

    def get_board_state(self):
        # ニューラルネットワーク用の盤面入力データを作成する
        state = np.zeros((2, 8, 8), dtype=np.float32)
        state[0] = (self.board == self.current_player).astype(np.float32)
        state[1] = (self.board == -self.current_player).astype(np.float32)
        return state

    def copy(self):
        # 現在の環境のコピーを作成する（探索用）
        new_env = OthelloEnv()
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        return new_env

# --- Minimax Depth 7 Logic ---
WEIGHTS = np.array([
    [ 120, -20,  20,   5,   5,  20, -20, 120],
    [ -20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [  20,  -5,  15,   3,   3,  15,  -5,  20],
    [   5,  -5,   3,   3,   3,   3,  -5,   5],
    [   5,  -5,   3,   3,   3,   3,  -5,   5],
    [  20,  -5,  15,   3,   3,  15,  -5,  20],
    [ -20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 120, -20,  20,   5,   5,  20, -20, 120],
])

def evaluate(board, player):
    # 盤面の評価値を計算する
    return np.sum(WEIGHTS[board == player]) - np.sum(WEIGHTS[board == -player])

def minimax(env, depth, alpha, beta, player, maximizing):
    # Minimax法による探索を行う
    if depth == 0: return evaluate(env.board, maximizing)
    valid = env.get_valid_moves(player)
    if not valid:
        if not env.get_valid_moves(-player):
            b, w = np.sum(env.board==1), np.sum(env.board==-1)
            if b > w: return 10000 if maximizing==1 else -10000
            elif w > b: return 10000 if maximizing==-1 else -10000
            return 0
        return minimax(env, depth, alpha, beta, -player, maximizing)

    best = -float('inf') if player == maximizing else float('inf')
    for m in valid:
        new_env = env.copy()
        new_env.make_move(m, player)
        val = minimax(new_env, depth-1, alpha, beta, -player, maximizing)
        if player == maximizing:
            best = max(best, val)
            alpha = max(alpha, val)
            if beta <= alpha: break
        else:
            best = min(best, val)
            beta = min(beta, val)
            if beta <= alpha: break
    return best

def get_teacher_moves(env, depth=7):
    # Depth 7探索による最善手（教師データ）を取得する
    valid = env.get_valid_moves(env.current_player)
    if not valid: return []
    
    best_moves = []
    best_val = -float('inf')
    
    for m in valid:
        new_env = env.copy()
        new_env.make_move(m, env.current_player)
        val = minimax(new_env, depth-1, -float('inf'), float('inf'), -env.current_player, env.current_player)
        
        if val > best_val:
            best_val = val
            best_moves = [m] 
        elif val == best_val:
            best_moves.append(m) 
            
    return best_moves

# --- ニューラルネットワークエージェント ---
class DeepOthelloModel(nn.Module):
    # 学習済みモデルの構造定義（10層CNN）
    def __init__(self):
        super(DeepOthelloModel, self).__init__()
        layers = [nn.Conv2d(2, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()]
        for _ in range(9):
            layers.extend([nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()])
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128*64, 1024), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(1024, 64)
        )
    def forward(self, x): return self.classifier(self.features(x))

def get_nn_move(model, env, device):
    # ニューラルネットワークを用いて次の一手を決定する
    valid = env.get_valid_moves(env.current_player)
    if not valid: return None
    state = torch.tensor(env.get_board_state()).unsqueeze(0).to(device)
    with torch.no_grad(): logits = model(state).cpu().numpy()[0]
    best_move = None
    best_logit = -float('inf')
    for m in valid:
        idx = m[0]*8 + m[1]
        if logits[idx] > best_logit:
            best_logit = logits[idx]
            best_move = m
    return best_move

def get_depth3_move(env):
    # 比較対象のDepth 3探索AIの手を取得する
    moves = get_teacher_moves(env, depth=3)
    if moves:
        import random
        return random.choice(moves)
    return None

# --- 1. 収録パート ---
def record_game(pattern, model, device):
    # AI同士の対戦を行い、棋譜データを記録する
    print("【収録開始】対戦データを生成中...")
    env = OthelloEnv()
    history = []
    progress = tqdm(total=60) 
    
    while True:
        valid = env.get_valid_moves(env.current_player)
        if not valid:
            if not env.get_valid_moves(-env.current_player): break
            env.current_player *= -1
            continue

        frame = {
            "board": env.board.copy(),
            "player": env.current_player,
            "valid_moves": valid,
            "teacher_moves": [], 
            "actual_move": None
        }

        frame["teacher_moves"] = get_teacher_moves(env, depth=7)

        move = None
        if pattern == 1:
            if env.current_player == 1: move = get_nn_move(model, env, device)
            else: move = get_depth3_move(env)
        else:
            move = get_nn_move(model, env, device)

        frame["actual_move"] = move
        history.append(frame)

        env.make_move(move, env.current_player)
        env.current_player *= -1
        progress.update(1)

    progress.close()
    
    history.append({
        "board": env.board.copy(),
        "player": 0, "valid_moves": [], "teacher_moves": [], "actual_move": None
    })
    
    b, w = np.sum(env.board == 1), np.sum(env.board == -1)
    print(f"\n収録完了！ 結果: 黒 {b} - 白 {w}")
    return history

# --- 2. 動画保存パート ---
def save_video(history):
    # 記録された棋譜データからmp4動画を生成する
    print(f"【動画生成】'{OUTPUT_VIDEO_FILE}' に保存しています...")
    
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE), pygame.HIDDEN)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, float(VIDEO_FPS), (WINDOW_SIZE, WINDOW_SIZE))
    
    if not out.isOpened():
        print("エラー: 動画ファイルを作成できませんでした。")
        return

    for i, frame in enumerate(tqdm(history, desc="Rendering")):
        screen.fill(COLOR_BG)
        
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, COLOR_LINE, rect, 1)

        board = frame["board"]
        valid_moves = frame["valid_moves"]
        teacher_moves = frame["teacher_moves"]

        for r in range(8):
            for c in range(8):
                if board[r, c] != 0:
                    color = COLOR_BLACK if board[r, c] == 1 else COLOR_WHITE
                    center = (c*CELL_SIZE + CELL_SIZE//2, r*CELL_SIZE + CELL_SIZE//2)
                    pygame.draw.circle(screen, color, center, CELL_SIZE//2 - 4)

        for (r, c) in valid_moves:
            center = (c*CELL_SIZE + CELL_SIZE//2, r*CELL_SIZE + CELL_SIZE//2)
            pygame.draw.circle(screen, COLOR_VALID, center, 8) 

        for (tr, tc) in teacher_moves:
            center = (tc*CELL_SIZE + CELL_SIZE//2, tr*CELL_SIZE + CELL_SIZE//2)
            pygame.draw.circle(screen, COLOR_GOLD, center, CELL_SIZE//2 - 2, 4) 

        pygame.display.flip()
        
        view = pygame.surfarray.array3d(screen)
        view = view.transpose([1, 0, 2])
        frame_img = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        
        repeat = FRAMES_PER_MOVE
        if i == len(history) - 1:
            repeat = VIDEO_FPS * 3 
            
        for _ in range(repeat):
            out.write(frame_img)

    out.release()
    pygame.quit()
    print("動画の保存が完了しました！")

# --- メイン実行 ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = DeepOthelloModel().to(device)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("モデルをロードしました。")
    except Exception as e:
        print(f"モデルロードエラー: {e}")
        return

    if PATTERN == 1: print(">>> パターン1: NN (黒) vs Depth3 (白)")
    else: print(">>> パターン2: NN (黒) vs NN (白)")

    history_data = record_game(PATTERN, model, device)
    save_video(history_data)

if __name__ == "__main__":
    main()
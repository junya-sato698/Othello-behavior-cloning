import numpy as np
import os
import time
import itertools 
from tqdm import tqdm
from othello8x8_logic_speed import Othello8x8EnvSpeed

# --- 設定 ---
DATA_FILE = "othello_infinite_depth7.npz"
CPU_DEPTH = 7   
RANDOM_PROBABILITY = 0.1  
# --- 賢い評価関数 (重み付け) ---
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

def evaluate_board(board, player):
    p_score = np.sum(WEIGHTS[board == player])
    o_score = np.sum(WEIGHTS[board == -player])
    return p_score - o_score

# --- 高速化Minimax ---
def minimax_speed(env, depth, alpha, beta, player, maximizing_player):
    if depth == 0:
        return evaluate_board(env.board, maximizing_player)

    valid_moves = env.get_valid_moves(player)
    
    if not valid_moves:
        if not env.get_valid_moves(-player):
            b_sum = np.sum(env.board == 1)
            w_sum = np.sum(env.board == -1)
            if b_sum > w_sum: return 10000 if maximizing_player == 1 else -10000
            if w_sum > b_sum: return 10000 if maximizing_player == -1 else -10000
            return 0
        return minimax_speed(env, depth, alpha, beta, -player, maximizing_player)

    if player == maximizing_player:
        max_eval = -np.inf
        for move in valid_moves:
            flips = env.make_move(move, player)
            eval = minimax_speed(env, depth - 1, alpha, beta, -player, maximizing_player)
            env.undo_move(move, flips, player)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = np.inf
        for move in valid_moves:
            flips = env.make_move(move, player)
            eval = minimax_speed(env, depth - 1, alpha, beta, -player, maximizing_player)
            env.undo_move(move, flips, player)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha: break
        return min_eval

# --- 実行ラッパー ---
def find_best_move(env, player, depth):
    valid_moves = env.get_valid_moves(player)
    if not valid_moves: return None
    
    best_move = None
    best_score = -np.inf
    np.random.shuffle(valid_moves)
    
    for move in valid_moves:
        flips = env.make_move(move, player)
        score = minimax_speed(env, depth - 1, -np.inf, np.inf, -player, player)
        env.undo_move(move, flips, player)
        
        if score > best_score:
            best_score = score
            best_move = move
            
    return best_move

# --- データ保存 ---
def save_data(states, actions):
    if len(states) == 0: return 0
    new_states = np.array(states, dtype=np.float32)
    new_actions = np.array(actions, dtype=np.int64)
    
    # 既存ファイルがあれば読み込んで結合
    if os.path.exists(DATA_FILE):
        try:
            data = np.load(DATA_FILE)
            old_states = data['states']
            old_actions = data['actions']
            final_states = np.concatenate((old_states, new_states), axis=0)
            final_actions = np.concatenate((old_actions, new_actions), axis=0)
        except:
            final_states = new_states
            final_actions = new_actions
    else:
        final_states = new_states
        final_actions = new_actions
        
    np.savez_compressed(DATA_FILE, states=final_states, actions=final_actions)
    return len(final_states)

# --- メイン処理 ---
def main():
    print(f"=== 無限データ収集モード (Depth {CPU_DEPTH}) ===")
    print(f"保存先: {DATA_FILE}")
    print(f"終了方法: キーボードの [Ctrl + C] を押してください")
    print(f"※中断しても、そこまでのデータは自動的に保存されます")
    
    buffer_states = []
    buffer_actions = []
    start_time = time.time()
    
    try:
        for game_i in tqdm(itertools.count(start=1), desc="Games"):
            env = Othello8x8EnvSpeed()
            game_over = False
            
            while True:
                current_player = env.current_player
                valid_moves = env.get_valid_moves(current_player)
                
                if not valid_moves and not env.get_valid_moves(-current_player):
                    break
                
                if not valid_moves:
                    env.switch_player()
                    continue
                
                move = None
                if np.random.rand() < RANDOM_PROBABILITY:
                    move = valid_moves[np.random.randint(len(valid_moves))]
                else:
                    move = find_best_move(env, current_player, CPU_DEPTH)
                
                # データ記録
                buffer_states.append(env.get_board_state())
                buffer_actions.append(env.get_int_from_action(move))
                
                # 実行
                env.make_move(move, current_player)
                env.switch_player()
            
            # 5試合ごとにこまめに保存 (長時間稼働時のリスクヘッジ)
            if game_i % 5 == 0:
                save_data(buffer_states, buffer_actions)
                buffer_states = [] 
                buffer_actions = []
                
    except KeyboardInterrupt:
        print("\n\nユーザーによる中断を検知しました。")
        print("終了処理を行っています... 電源を切らないでください。")
    finally:
        # 最後にバッファに残っているデータを保存
        total = save_data(buffer_states, buffer_actions)
        elapsed = time.time() - start_time
        print(f"\n=== 収集完了 ===")
        print(f"総データ数: {total} 手")
        print(f"稼働時間: {elapsed/3600:.2f} 時間")

if __name__ == "__main__":
    main()
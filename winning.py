import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import os

# --- 必須: ゲームロジックのインポート ---
try:
    from othello8x8_logic_speed import Othello8x8EnvSpeed
except ImportError:
    print("★エラー: 'othello8x8_logic_speed.py' が見つかりません。")
    exit()

# --- 実験設定 ---
DATA_FILE = "othello_infinite_depth7.npz"
MODEL_BASE_NAME = "bc_model_fraction"
DATA_FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0] 

# ハイパーパラメータ
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
INPUT_CHANNELS = 2
N_ACTIONS = 64
NUM_FILTERS = 128

# 対戦設定
EVAL_GAMES = 100
OPPONENT_DEPTH = 3
TEACHER_DEPTH = 7
TEACHER_RANDOMNESS = 0.1

# --- 思考ロジック (スクリプト内に内蔵) ---

# 評価重みテーブル
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

def evaluate_board_score(board, player):
    # 盤面の評価値（重み付き石差）を計算する
    p_score = np.sum(WEIGHTS[board == player])
    o_score = np.sum(WEIGHTS[board == -player])
    return p_score - o_score

def minimax_logic(env, depth, alpha, beta, player, maximizing_player):
    # Alpha-Beta法を用いたMinimax探索を行う
    if depth == 0:
        return evaluate_board_score(env.board, maximizing_player)

    valid_moves = env.get_valid_moves(player)
    
    if not valid_moves:
        if not env.get_valid_moves(-player):
            b_sum = np.sum(env.board == 1)
            w_sum = np.sum(env.board == -1)
            if b_sum > w_sum: return 10000 if maximizing_player == 1 else -10000
            if w_sum > b_sum: return 10000 if maximizing_player == -1 else -10000
            return 0
        return minimax_logic(env, depth, alpha, beta, -player, maximizing_player)

    if player == maximizing_player:
        max_eval = -float('inf')
        for move in valid_moves:
            flips = env.make_move(move, player)
            eval = minimax_logic(env, depth - 1, alpha, beta, -player, maximizing_player)
            env.undo_move(move, flips, player)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = float('inf')
        for move in valid_moves:
            flips = env.make_move(move, player)
            eval = minimax_logic(env, depth - 1, alpha, beta, -player, maximizing_player)
            env.undo_move(move, flips, player)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha: break
        return min_eval

def find_best_move_logic(env, player, depth):
    # 指定された深さで探索を行い、最善手を選択する
    valid_moves = env.get_valid_moves(player)
    if not valid_moves: return None
    
    best_move = None
    best_score = -float('inf')
    
    indices = np.arange(len(valid_moves))
    np.random.shuffle(indices)
    
    for i in indices:
        move = valid_moves[i]
        flips = env.make_move(move, player)
        score = minimax_logic(env, depth - 1, -float('inf'), float('inf'), -player, player)
        env.undo_move(move, flips, player)
        
        if score > best_score:
            best_score = score
            best_move = move
            
    return best_move

# --- データセット定義 ---
class OthelloExpertDataset(Dataset):
    # 学習用データを読み込み、前処理とデータ拡張を行うクラス
    def __init__(self, data_file, transform=False):
        self.transform = transform
        if not os.path.exists(data_file):
            print(f"エラー: '{data_file}' なし")
            self.states = torch.zeros(100, 2, 8, 8); self.actions = torch.zeros(100, dtype=torch.long)
        else:
            data = np.load(data_file)
            self.states = torch.tensor(data['states'], dtype=torch.float32)
            self.actions = torch.tensor(data['actions'], dtype=torch.long)
    def __len__(self):
        return len(self.states) * 8 if self.transform else len(self.states)
    def __getitem__(self, idx):
        if self.transform:
            original_idx, aug_mode = idx // 8, idx % 8
            state, action_idx = self.states[original_idx], self.actions[original_idx].item()
            return self.apply_augment(state, action_idx, aug_mode)
        return self.states[idx], self.actions[idx]
    def apply_augment(self, state, action_idx, mode):
        r, c = action_idx // 8, action_idx % 8
        if mode >= 4:
            state = torch.flip(state, [2]); c = 7 - c; mode -= 4
        k = mode
        if k > 0:
            state = torch.rot90(state, k, [1, 2])
            for _ in range(k): r, c = 7 - c, r
        return state, torch.tensor(r * 8 + c, dtype=torch.long)

# --- モデル定義 ---
class DeepOthelloModel(nn.Module):
    # 10層の畳み込みニューラルネットワークを定義するクラス
    def __init__(self, input_channels=INPUT_CHANNELS, n_actions=N_ACTIONS, filters=NUM_FILTERS):
        super(DeepOthelloModel, self).__init__()
        layers = [nn.Conv2d(input_channels, filters, 3, padding=1), nn.BatchNorm2d(filters), nn.ReLU()]
        for _ in range(9):
            layers.extend([nn.Conv2d(filters, filters, 3, padding=1), nn.BatchNorm2d(filters), nn.ReLU()])
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(filters * 8 * 8, 1024), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(1024, n_actions)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# --- 評価用関数 ---
def evaluate_acc(model, dataloader, device):
    # 検証データに対するモデルの正解率を計算する
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for states, actions in dataloader:
            states, actions = states.to(device), actions.to(device)
            preds = torch.argmax(model(states), dim=1)
            correct += (preds == actions).sum().item()
            total += states.size(0)
    return (correct / total) * 100

def run_match_benchmark(black_player_func, white_player_func, num_games, desc):
    # 2つのエージェントを対戦させ、勝率を測定する
    wins_black = 0
    draws = 0
    for i in tqdm(range(num_games), desc=desc, leave=True):
        env = Othello8x8EnvSpeed()
        while True:
            valid_moves = env.get_valid_moves(env.current_player)
            if not valid_moves:
                if not env.get_valid_moves(-env.current_player): break
                env.switch_player()
                continue
            
            if env.current_player == 1:
                move = black_player_func(env)
            else:
                move = white_player_func(env)
            env.make_move(move, env.current_player)
            env.switch_player()
            
        is_over, winner = env.get_game_status()
        if winner == 1: wins_black += 1
        elif winner == 0: draws += 1
            
    return (wins_black / num_games) * 100

# --- エージェント定義 ---
def make_model_agent(model, device):
    # ニューラルネットワークを用いて手を決定するエージェントを作成する
    def agent(env):
        state = env.get_board_state()
        state_tensor = torch.tensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(state_tensor).squeeze(0).cpu().numpy()
        valid_moves = env.get_valid_moves(env.current_player)
        if not valid_moves: return None
        mask = np.full(64, -float('inf'))
        for vm in valid_moves:
            mask[env.get_int_from_action(vm)] = 0
        best_idx = np.argmax(logits + mask)
        return env.get_action_from_int(best_idx)
    return agent

def make_minimax_agent(depth):
    # Minimax探索を用いて手を決定するエージェントを作成する
    def agent(env):
        return find_best_move_logic(env, env.current_player, depth)
    return agent

def make_teacher_agent_with_random(depth, random_prob):
    # 一定確率でランダムな手を指す教師エージェントを作成する
    def agent(env):
        valid_moves = env.get_valid_moves(env.current_player)
        if not valid_moves: return None
        if np.random.rand() < random_prob:
            return valid_moves[np.random.randint(len(valid_moves))]
        return find_best_move_logic(env, env.current_player, depth)
    return agent

# --- メイン実験ループ ---
def run_experiment():
    # データ量を変化させながらモデルを学習し、性能を評価する
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")
    
    print(f"\n【基準測定】教師AI (Depth {TEACHER_DEPTH}, 10% Random) vs Depth {OPPONENT_DEPTH}")
    teacher_agent = make_teacher_agent_with_random(TEACHER_DEPTH, TEACHER_RANDOMNESS)
    opponent_agent = make_minimax_agent(OPPONENT_DEPTH)
    
    rate_b = run_match_benchmark(teacher_agent, opponent_agent, 50, "Teacher(Black)")
    rate_w = 100 - run_match_benchmark(opponent_agent, teacher_agent, 50, "Teacher(White)")
    teacher_avg_rate = (rate_b + rate_w) / 2
    print(f"★ 教師AI平均勝率: {teacher_avg_rate:.1f}%")

    full_dataset = OthelloExpertDataset(DATA_FILE)
    total_len = len(full_dataset)
    indices = list(range(total_len))
    np.random.shuffle(indices)
    
    n_train_full = int(0.8 * total_len)
    n_val = int(0.1 * total_len)
    base_train_indices = indices[:n_train_full]
    val_indices = indices[n_train_full : n_train_full + n_val]
    
    val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False)
    
    class OthelloAugmentedSubset(Dataset):
        def __init__(self, dataset, indices): self.dataset, self.indices = dataset, indices
        def __len__(self): return len(self.indices) * 8
        def __getitem__(self, idx):
            aug, ptr = idx % 8, idx // 8
            st, act = self.dataset.states[self.indices[ptr]], self.dataset.actions[self.indices[ptr]].item()
            return self.dataset.apply_augment(st, act, aug)

    results_acc = []
    results_win_rate = []

    for frac in DATA_FRACTIONS:
        print(f"\n=== データ割合 {frac*100}% ===")
        current_n = int(n_train_full * frac)
        train_loader = DataLoader(OthelloAugmentedSubset(full_dataset, base_train_indices[:current_n]), batch_size=BATCH_SIZE, shuffle=True)
        print(f"学習データ数: {len(train_loader.dataset)}")
        
        model = DeepOthelloModel(filters=NUM_FILTERS).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
        
        best_acc = 0.0
        patience_cnt = 0
        save_path = f"{MODEL_BASE_NAME}_{frac}.pth"
        
        for epoch in range(MAX_EPOCHS):
            model.train()
            for states, actions in train_loader:
                states, actions = states.to(device), actions.to(device)
                optimizer.zero_grad()
                loss = criterion(model(states), actions)
                loss.backward()
                optimizer.step()
            
            acc = evaluate_acc(model, val_loader, device)
            scheduler.step(acc)
            
            if acc > best_acc:
                best_acc = acc
                patience_cnt = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE: break
        
        results_acc.append(best_acc)
        print(f"Best Acc: {best_acc:.2f}%")
        
        model.load_state_dict(torch.load(save_path))
        model_agent = make_model_agent(model, device)
        wr_black = run_match_benchmark(model_agent, opponent_agent, 50, "Model(Black)")
        wr_white = 100 - run_match_benchmark(opponent_agent, model_agent, 50, "Model(White)")
        avg_wr = (wr_black + wr_white) / 2
        results_win_rate.append(avg_wr)
        print(f"Win Rate: {avg_wr:.1f}%")

    print("\n" + "="*40)
    print("=== 実験結果 (グラフ作成用データ) ===")
    print("="*40)
    print(f"Data Fractions: {DATA_FRACTIONS}")
    print(f"Valid Accuracies: {results_acc}")
    print(f"Win Rates (vs Depth3): {results_win_rate}")
    print(f"Teacher Strength (vs Depth3): {teacher_avg_rate:.2f}%")
    print("実験完了！")

if __name__ == "__main__":
    run_experiment()
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import os

# --- ハイパーパラメータ ---
DATA_FILE = "othello_infinite_depth7.npz" 
MODEL_SAVE_PATH = "bc_model_deep_cnn_best.pth" 
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4  
MAX_EPOCHS = 100 
PATIENCE = 10 

INPUT_CHANNELS = 2
N_ACTIONS = 64
NUM_FILTERS = 128    

# --- データセットクラス (共通) ---
class OthelloExpertDataset(Dataset):
    def __init__(self, data_file, transform=False):
        self.transform = transform
        print(f"'{data_file}' からデータをロード中... (Data Augmentation: {self.transform})")
        if not os.path.exists(data_file):
            print(f"エラー: '{data_file}' が見つかりません。")
            self.states = torch.zeros(100, 2, 8, 8)
            self.actions = torch.zeros(100, dtype=torch.long)
        else:
            try:
                data = np.load(data_file)
                self.states = torch.tensor(data['states'], dtype=torch.float32)
                self.actions = torch.tensor(data['actions'], dtype=torch.long)
                print(f"ロード完了。基本データ数: {len(self.states)}")
            except Exception as e:
                print(f"データ読み込みエラー: {e}")
                exit()
            
    def __len__(self):
        if self.transform:
            return len(self.states) * 8
        else:
            return len(self.states)
    
    def __getitem__(self, idx):
        if self.transform:
            original_idx = idx // 8
            aug_mode = idx % 8
            state = self.states[original_idx]
            action_idx = self.actions[original_idx].item()
            state, action_idx = self.apply_augment(state, action_idx, aug_mode)
            return state, action_idx
        else:
            return self.states[idx], self.actions[idx]

    def apply_augment(self, state, action_idx, mode):
        r, c = action_idx // 8, action_idx % 8
        if mode >= 4:
            state = torch.flip(state, [2]) 
            c = 7 - c 
            mode -= 4
        k = mode
        if k > 0:
            state = torch.rot90(state, k, [1, 2]) 
            for _ in range(k):
                new_r = 7 - c
                new_c = r
                r, c = new_r, new_c
        new_action_idx = r * 8 + c
        return state, torch.tensor(new_action_idx, dtype=torch.long)

# ---  モデル定義: Deep Plain CNN ---
class DeepOthelloModel(nn.Module): 
    def __init__(self, input_channels=INPUT_CHANNELS, n_actions=N_ACTIONS, filters=NUM_FILTERS): 
        super(DeepOthelloModel, self).__init__() 
        

        layers = []
        
        # 1層目
        layers.append(nn.Conv2d(input_channels, filters, kernel_size=3, padding=1)) 
        layers.append(nn.BatchNorm2d(filters)) 
        layers.append(nn.ReLU()) 
        
        # 2〜10層目 
        for _ in range(9): 
            layers.append(nn.Conv2d(filters, filters, kernel_size=3, padding=1)) 
            layers.append(nn.BatchNorm2d(filters)) 
            layers.append(nn.ReLU())
            
        # 特徴抽出部をまとめる
        self.features = nn.Sequential(*layers) 
        
        # 分類ヘッド (全結合層)
        self.classifier = nn.Sequential( 
            nn.Flatten(),
            nn.Linear(filters * 8 * 8, 1024), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(1024, n_actions) 
        )

    def forward(self, x): 
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 評価用関数 ---
def evaluate(model, dataloader, criterion, device):
    model.eval() 
    total_loss = 0.0 
    correct_predictions = 0 
    total_samples = 0 
    
    with torch.no_grad(): 
        for states, actions in dataloader:
            states = states.to(device) 
            actions = actions.to(device) 
            
            logits = model(states) 
            loss = criterion(logits, actions) 
            
            total_loss += loss.item() * states.size(0) 
            preds = torch.argmax(logits, dim=1) 
            correct_predictions += (preds == actions).sum().item() 
            total_samples += states.size(0) 
            
    avg_loss = total_loss / total_samples 
    accuracy = (correct_predictions / total_samples) * 100 
    return avg_loss, accuracy 

# --- メイン学習ループ ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device} を使用します。")
    
    # データセット準備
    temp_dataset = OthelloExpertDataset(DATA_FILE)
    total_size = len(temp_dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)
    
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]
    
    # Subsetラッパー (学習データ拡張用)
    class OthelloAugmentedSubset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices) * 8
        def __getitem__(self, idx):
            aug_mode = idx % 8
            original_idx_ptr = idx // 8
            real_data_idx = self.indices[original_idx_ptr]
            state = self.dataset.states[real_data_idx]
            action_idx = self.dataset.actions[real_data_idx].item()
            state, action_idx = self.dataset.apply_augment(state, action_idx, aug_mode)
            return state, action_idx

    base_dataset = OthelloExpertDataset(DATA_FILE, transform=False)
    train_dataset = OthelloAugmentedSubset(base_dataset, train_indices)
    val_dataset = Subset(base_dataset, val_indices)
    test_dataset = Subset(base_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"拡張後の学習データ数: {len(train_dataset)} (x8倍)")
    
    # モデル初期化
    model = DeepOthelloModel(filters=NUM_FILTERS).to(device)
    print("DeepOthelloModel (10層CNN) を構築しました。")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    best_val_accuracy = 0.0
    patience_counter = 0
    
    print(f"\n--- 訓練開始 (Deep CNN + Augmentation, Patience: {PATIENCE}) ---")
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=True)
        
        for states, actions in loop:
            states = states.to(device)
            actions = actions.to(device)
            
            logits = model(states)
            loss = criterion(logits, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * states.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == actions).sum().item()
            train_total += states.size(0)
            
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / train_total
        train_acc = (train_correct / train_total) * 100
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1} 結果:")
        print(f"  [Train] Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  [Valid] Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_accuracy:
            print(f"  ★ ベスト更新！ ({best_val_accuracy:.2f}% -> {val_acc:.2f}%)")
            best_val_accuracy = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            print(f"  記録更新ならず... ({patience_counter}/{PATIENCE})")
            
            if patience_counter >= PATIENCE:
                print(f"\n✋ Early Stopping")
                break
    
    print("\n--- 最終テスト評価 ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"モデル '{MODEL_SAVE_PATH}' のテスト結果:")
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    train()
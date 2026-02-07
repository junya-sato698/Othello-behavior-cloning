import numpy as np 

class Othello8x8EnvSpeed:
    # 8x8オセロのゲームロジックと盤面状態を管理するクラス
    def __init__(self): 
        # 初期化：盤面作成、初期配置、手番、方向ベクトルの定義
        self.board_size = 8 
        self.board = np.zeros((self.board_size, self.board_size), dtype=int) 
        mid = self.board_size // 2 
        self.board[mid - 1, mid - 1] = 1 
        self.board[mid, mid] = 1 
        self.board[mid - 1, mid] = -1 
        self.board[mid, mid - 1] = -1 
        self.current_player = 1 
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def get_board_state(self): 
        # ニューラルネットワーク入力用の盤面状態（2ch: 自石, 他石）を返す
        state = np.zeros((2, self.board_size, self.board_size), dtype=np.float32) 
        state[0, self.board == self.current_player] = 1.0 
        state[1, self.board == -self.current_player] = 1.0 
        return state 

    def get_valid_moves(self, player): 
        # 指定されたプレイヤーの有効手（着手可能な座標）のリストを取得する
        valid_moves = [] 
        for r in range(self.board_size): 
            for c in range(self.board_size): 
                if self.board[r, c] == 0: 
                    if self._check_flips(r, c, player): 
                        valid_moves.append((r, c)) 
        return valid_moves 

    def _get_flips(self, r, c, player):
        # 指定された位置に置いた際に裏返る石の座標リストを取得（make_move用）
        flips = [] 
        opponent = -player 
        for dr, dc in self.directions: 
            cr, cc = r + dr, c + dc 
            potential_flips = [] 
            while 0 <= cr < self.board_size and 0 <= cc < self.board_size: 
                if self.board[cr, cc] == opponent: 
                    potential_flips.append((cr, cc)) 
                elif self.board[cr, cc] == player: 
                    flips.extend(potential_flips) 
                    break 
                else: 
                    break 
                cr, cc = cr + dr, cc + dc 
        return flips 

    def _check_flips(self, r, c, player): 
        # 裏返る石があるかだけを高速に判定（get_valid_moves用）
        opponent = -player 
        for dr, dc in self.directions: 
            cr, cc = r + dr, c + dc 
            has_opponent = False 
            while 0 <= cr < self.board_size and 0 <= cc < self.board_size: 
                if self.board[cr, cc] == opponent: 
                    has_opponent = True 
                elif self.board[cr, cc] == player: 
                    if has_opponent: return True 
                    break 
                else:
                    break 
                cr, cc = cr + dr, cc + dc 
        return False 

    def make_move(self, move, player):
        # 石を置き、挟んだ相手の石を裏返す（戻り値として裏返った石のリストを返す）
        r, c = move 
        flips = self._get_flips(r, c, player) 
        if not flips: return None 

        self.board[r, c] = player 
        for fr, fc in flips: 
            self.board[fr, fc] = player 
        
        return flips 

    def undo_move(self, move, flips, player):
        # 探索用に盤面を1手前の状態に戻す（Undo処理）
        r, c = move 
        self.board[r, c] = 0 
        
        opponent = -player 
        for fr, fc in flips: 
            self.board[fr, fc] = opponent 

    def switch_player(self):
        # 手番を交代する
        self.current_player = -self.current_player 
        
    def get_game_status(self):
        # 簡易的なゲーム終了判定と勝敗判定を行う
        b_count = np.sum(self.board == 1) 
        w_count = np.sum(self.board == -1) 
        
        if b_count + w_count == 64 or b_count == 0 or w_count == 0: 
             if b_count > w_count: return True, 1 
             if w_count > b_count: return True, -1 
             return True, 0 
        
        return False, 0

    def get_action_from_int(self, action_int):
        # 整数インデックス(0-63)を座標(r, c)に変換する
        return (action_int // self.board_size, action_int % self.board_size) 
    
    def get_int_from_action(self, action_tuple):
        # 座標(r, c)を整数インデックス(0-63)に変換する
        return action_tuple[0] * self.board_size + action_tuple[1]
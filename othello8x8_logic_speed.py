import numpy as np 

class Othello8x8EnvSpeed: #8x8オセロのゲームロジックを管理するクラス
    def __init__(self): 
        self.board_size = 8 #盤面のサイズ(8)の定義
        self.board = np.zeros((self.board_size, self.board_size), dtype=int) #8x8の0行列の作成しその行列内に整数のみ入ることの定義(白黒無しのすべて整数のため)
        mid = self.board_size // 2 #ボードの中心の定義(4)
        self.board[mid - 1, mid - 1] = 1 #(3,3)に黒を配置
        self.board[mid, mid] = 1 #(4,4)に黒を配置
        self.board[mid - 1, mid] = -1 #(3,4)に白を配置
        self.board[mid, mid - 1] = -1 #(4,3)に白を配置
        self.current_player = 1 #今のプレイヤーがどっちか定義(最初は黒)
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        #8方向のチェッカー用(右,下,左,上,右下,右上,左下,左上)


    def get_board_state(self): #ニューラルネットワーク入力用の盤面状態を返す (2, 8, 8)
        state = np.zeros((2, self.board_size, self.board_size), dtype=np.float32) #0テンソルの作成(2ch,8行,8列)
        state[0, self.board == self.current_player] = 1.0 #0チャンネルは自分の石が１，それ以外は０のまま
        state[1, self.board == -self.current_player] = 1.0 #1チャンネルは相手の石が１，それ以外は０のまま
        return state #上記をした環境を返す

    def get_valid_moves(self, player): #指定されたプレイヤーの有効な手をすべて見つける
        valid_moves = [] #有効手の格納リストの初期化
        for r in range(self.board_size): #行のループ
            for c in range(self.board_size): #列のループ(上と合わせて盤面上すべてを探索)
                if self.board[r, c] == 0: #空きマスなら下のコードへ、空いてないなら次の座標へ
                    if self._check_flips(r, c, player): # 高速化のためチェックのみ
                        valid_moves.append((r, c)) #空きマスかつひっくりかえせるので有効手のリストに格納
        return valid_moves #有効手リストを返す

    # フリップする石のリストを返す（make_move用）
    def _get_flips(self, r, c, player):
        flips = [] #ひっくり返す石のリストの初期化
        opponent = -player #探すべき石の定義(相手の石のみ)
        for dr, dc in self.directions: #定義済みの方向への繰り返し
            cr, cc = r + dr, c + dc #指定された方向へ１つ進む
            potential_flips = [] #ひっくりかえせる可能性のある石として一時的に保持するリストの初期化
            while 0 <= cr < self.board_size and 0 <= cc < self.board_size: #盤面外に出るまでループ
                if self.board[cr, cc] == opponent: #もし相手の石を発見したら下の行へ
                    potential_flips.append((cr, cc)) #potential_flipsに座標を入れて６行下へ
                elif self.board[cr, cc] == player: #もし自分の石を発見したら下の行へ
                    flips.extend(potential_flips) #potential_flipsにためていた座標をflipsに入れ替えて終了
                    break   #ループを抜ける
                else:   #どっちにも当てはまらないならそのまま終了
                    break   
                cr, cc = cr + dr, cc + dc #次のマスへ進む
        return flips  #裏返す石のリストを返す

    # フリップがあるかだけ確認（get_valid_moves用・高速）
    def _check_flips(self, r, c, player): 
        opponent = -player #探すべき石の定義(相手の石のみ)
        for dr, dc in self.directions: #定義済みの方向への繰り返し
            cr, cc = r + dr, c + dc #指定された方向へ１つ進む
            has_opponent = False #相手の石を見つけたかどうかのフラグ
            while 0 <= cr < self.board_size and 0 <= cc < self.board_size: #盤面外に出るまでループ
                if self.board[cr, cc] == opponent: #もし相手の石を発見したら下の行へ
                    has_opponent = True #相手の石を見つけたフラグをTrueに
                elif self.board[cr, cc] == player: #もし自分の石を発見したら下の行へ
                    if has_opponent: return True #相手の石を見つけてから自分の石を見つけたならTrueで
                    break   #ループを抜ける
                else:
                    break   #ループを抜ける
                cr, cc = cr + dr, cc + dc #次のマスへ進む
        return False #どの方向にもフリップがなかった場合Falseを返す

    # ★重要：石を置き、裏返った石のリストを返す（Undoのため）
    def make_move(self, move, player):
        r, c = move #座標を取得
        flips = self._get_flips(r, c, player) # 裏返る石のリストを取得
        if not flips: return None # 無効な手

        self.board[r, c] = player
        for fr, fc in flips:
            self.board[fr, fc] = player
        
        return flips # 裏返った石の場所を返す

    # ★重要：手を戻す（Undo）
    def undo_move(self, move, flips, player):
        r, c = move
        self.board[r, c] = 0 # 置いた石を取り除く
        
        opponent = -player
        for fr, fc in flips:
            self.board[fr, fc] = opponent # 裏返った石を元に戻す

    def switch_player(self):
        self.current_player = -self.current_player
        
    def get_game_status(self):
        # 高速化のため簡易チェック
        b_count = np.sum(self.board == 1)
        w_count = np.sum(self.board == -1)
        # 盤面が埋まったか、どちらかの石がなくなった場合のみ終了とする簡易版
        if b_count + w_count == 64 or b_count == 0 or w_count == 0:
             if b_count > w_count: return True, 1
             if w_count > b_count: return True, -1
             return True, 0
        
        # パス判定を含む厳密な終了判定は重いので、
        # Minimaxの再帰内では「打つ手なし」で処理させる
        return False, 0

    # ★追加したメソッド 1
    def get_action_from_int(self, action_int):
        """ 0-63 の整数を (r, c) に変換 """
        return (action_int // self.board_size, action_int % self.board_size)
    
    # ★追加したメソッド 2
    def get_int_from_action(self, action_tuple):
        """ (r, c) を 0-63 の整数に変換 """
        return action_tuple[0] * self.board_size + action_tuple[1]
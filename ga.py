import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

class Ga:
  def __init__(self,arr,genes_num,cross_num):
    #扱う配列
    self.arr = arr
    self.col, self.row = arr.shape
    #スタートするx軸。行の中間からスタートする。
    self.start_point = self.col // 2
    #遺伝子数
    self.genes = genes_num
    #n世代のベクトル
    self.n_generation_vector = np.array([self.init_genes() for i in np.arange(self.genes)])
    #交叉ベクトル。個数
    self.cross_num = cross_num
    self.cross_vector = None

  def init_genes(self):
    # -1, 0, 1のどれかを入れたベクトルを作成する。
    while(True):
      vector =  np.random.choice([ -1, 0, 1 ],  self.row)
      sum = np.cumsum(vector) + self.start_point
      if np.all((0 <= sum) & (sum < self.col)):
        return vector

  def get_cross_vector(self,change_index_num):
    self.cross_vector = np.array([j for x in range(self.cross_num // 2) for j in self.cross(self.n_generation_vector,change_index_num)])

  def cross(self,vector,change_index_num):
      while(True):
          #遺伝子を2つランダムに選ぶ。
          vec = np.random.default_rng().choice( vector, size = 2, replace=False )
          #入れ替える値のインデックスをランダムに決める。
          index = np.random.choice(np.arange(self.row), size = change_index_num, replace=False)
          #スワップ
          swap = vec[:,index].copy()
          swap[[0, 1]] = swap[[1, 0]]
          vec[:,index] = swap
          #判定
          sum = np.cumsum(vec, axis = 1) + self.start_point
          if np.all((0 <= sum) & (sum < self.col)):
              return vec

  def road_plot(self,img_name = None):
    #通ったルートをプロットする。
    accumulation_arr = np.cumsum(self.n_generation_vector, axis = 1) + self.start_point
    road = np.apply_along_axis(lambda x: np.bincount(x, minlength = self.col), axis = 0, arr = accumulation_arr)
    plt.clf()
    return sns.heatmap(road, cmap='Blues').get_figure()

  def point(self):
    #点数を表示する
    n_gen_plus_cross = np.vstack([ self.n_generation_vector, self.cross_vector ])
    accumulation_arr = np.cumsum(n_gen_plus_cross, axis = 1) + self.start_point
    range = np.fromfunction(lambda i, j: j, accumulation_arr.shape, dtype=int)
    return np.sum( self.arr[accumulation_arr,range], axis=1 )
  
  def rank(self):
    #点数が高いものから、indexを返す。
    return np.argsort(self.point())[::-1]

  def select(self,probability_width = 1):
    #上位nつのインデックスを採用し、n_generation_vectorに渡す。
    n_gen_plus_cross = np.vstack([ self.n_generation_vector, self.cross_vector ])
    weight = np.linspace(probability_width, 1, self.genes + self.cross_num) / np.sum(np.linspace(probability_width, 1, self.genes + self.cross_num))
    self.n_generation_vector = n_gen_plus_cross[np.random.choice( self.rank(), size=self.genes, p=weight, replace=False)]



#ここを変更する
#ステージを読み込む
arr = np.loadtxt("stage1.csv", delimiter=',', dtype='int')
#遺伝子の数:15 交叉した遺伝子の数:30 入れ替えるベクトル数:5 ルーレット選択の重み:15
genes_num, cross_num, change_index_num, probability_width = 15, 30, 5, 15

#コンストラクタ
ga = Ga(arr, genes_num=genes_num, cross_num=cross_num)
generation = 1
while len(np.unique(ga.n_generation_vector, axis = 0)) != 1:
  ga.get_cross_vector(change_index_num = change_index_num)
  ga.select(probability_width = probability_width)
  generation += 1

print(f"{generation}世代 : 合計{ga.point()[0]}点")
ga.road_plot()
#plotを保存する場合
#ga.road_plot().savefig("img_name.png", dpi=400)

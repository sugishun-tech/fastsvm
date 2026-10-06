# CythonでSVMを自作する――SMO・線形座標降下・KKT診断まで備えた「fastsvm」

SVM を高速に使うだけなら、既存の十分に最適化されたライブラリがある。それでも自分で実装する理由を挙げるなら、「最適化の内部状態を、自分で検査できる形で手元に置きたい」からだ。

そこで、Cython から C にコンパイルする SVM プロジェクト **fastsvm** を作った。scikit-learn 風の API を持つが、学習を `sklearn.svm.SVC` などに委譲するラッパーではない。線形モデルの双対座標降下法と、非線形モデルの SMO を独自に実装している。

さらに、主双対ギャップ、KKT 残差、双対変数、解析的な入力勾配・Hessian、Gram 行列のスペクトルといった、数理的に中身を調べる機能も付けた。

実測では、今回の合成データ条件で RBF の二値分類・回帰の推論が sklearn の約 **3.5 倍**、dense な線形分類の学習が約 **1.55 倍**になった。一方、線形回帰や多クラス分類の学習で負けたケースもある。この記事では両方を掲載する。「世界最速」という記事ではなく、実装・数理・検証・測定条件を揃えて公開する記事である。

## 1. 実装したもの

| クラス | 学習法 | 用途 |
|---|---|---|
| `LinearSVC` | ランダム化双対座標降下法 | dense / CSR の線形分類、hinge / squared hinge、二値 / OvR |
| `LinearSVR` | soft-thresholding を使う双対座標降下法 | dense / CSR の線形回帰、通常 / 二乗 epsilon-insensitive 損失 |
| `SVC` | 二次情報による作業集合選択付き SMO | カーネル分類、二値 / OvO |
| `SVR` | 共通 SMO による epsilon-SVR | カーネル回帰 |
| `OneClassSVM` | 共通 SMO による正規化 one-class SVM | 外れ値検知 |

組み込みカーネルは linear、RBF、polynomial、sigmoid、Laplacian。precomputed Gram と callable にも対応し、カーネルの和・積・非負スカラー倍も記述できる。

大きなデータ用には `RBFSampler` と `Nystroem` を用意した。ただし、有限次元の近似特徴に置き換えることと、厳密なカーネル SVM を高速に解くことは区別する。

全体は CPU / float64 実装で、GPU や float32 専用ソルバーは含まない。Cython が担当するのは学習・カーネル計算のホットループであり、API、入力検証、スペクトル解析などは Python / NumPy / SciPy に分けている。

## 2. インストールと最初の分類

以下のコマンドは、ソースを展開した `fastsvm` のルートで実行する。公開 PyPI パッケージの存在を前提にした `pip install fastsvm` ではなく、手元のソースをインストールする手順である。

Debian / Linux の例:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[dev]'
```

既定では OpenMP を要求しない portable build になる。同じマシンで使用するための最適化ビルドは、**まだビルドしていないソースツリー**から次のように行う。

```bash
FASTSVM_NATIVE=1 FASTSVM_OPENMP=1 python -m pip install '.[dev]'
```

`FASTSVM_NATIVE=1` は GCC / Clang の `-march=native` を使う。その wheel を異なる命令セットの CPU に配布してはいけない。既にビルドしたツリーで設定を変更するときは、`build/` と生成された `src/fastsvm/_core.c` を削除してから再ビルドする。

コンパイルされた実際の OpenMP 対応状況は取得できる。

```python
from fastsvm import build_info
print(build_info())
# {'openmp': True または False, 'backend': 'Cython/C', 'precision': 'float64'}
```

通常の使い方は sklearn とほぼ同じである。

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from fastsvm import SVC

X, y = make_moons(n_samples=1000, noise=0.18, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=9
)

search = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    {"svc__C": [0.5, 2, 8], "svc__gamma": [0.2, 1]},
    cv=3,
)
search.fit(X_train, y_train)

print(search.best_params_)
print(search.score(X_test, y_test))
print(search.best_estimator_[-1].optimization_report())
```

標準化を交差検証の外でデータ全体に適用してから評価しない。`Pipeline` に入れることで、各 fold の学習データだけから前処理の統計を推定できる。

## 3. 線形と非線形を同じソルバーに押し込まない

構成は次のとおり。

```text
fastsvm/
├── src/fastsvm/
│   ├── _core.pyx          # Cython/C: 座標降下、SMO、LRU、カーネル
│   ├── _validation.py    # 形状・重み・パラメーターの検証
│   ├── linear.py         # LinearSVC / LinearSVR
│   ├── svm.py            # SVC / SVR / OneClassSVM
│   ├── kernels.py        # カーネル合成・スペクトル解析
│   ├── approximation.py  # RFF / Nyström
│   └── analysis.py       # C 掃引・margin 統計
├── tests/
├── benchmarks/
├── examples/
└── docs/
```

線形問題では重みベクトルを明示的に更新すればよく、全 Gram 行列は要らない。逆に、非線形カーネル問題は `w` を有限次元の元の特徴空間で保持できないので、双対変数とカーネル行を扱う。

この違いを無視して「何でも汎用 QP に渡す」と、利用できる構造を捨ててしまう。fastsvm では二系統に分けた。

## 4. 線形 SVM――一座標の更新を安くする

### 4.1 解いている主問題

二値ラベルを $y_i\in\{-1,+1\}$ とする。切片を学習する場合は、人工特徴 $s$ を追加する。

$$
\tilde x_i=(x_i,s),\qquad \tilde w=(w,b/s).
$$

通常の hinge loss を用いる主問題は、

$$
P=\frac12\|\tilde w\|^2+
\sum_i C_i\max(0,1-y_i\tilde w^T\tilde x_i).
$$

ここで $C_i=C\times\mathrm{sample\_weight}_i\times\mathrm{class\_weight}_i$。損失は平均ではなく **総和** である。

切片の正則化項は $b^2/(2s^2)$ になる。つまり `LinearSVC` と `SVC(kernel="linear")` は切片の扱いまで同じとは限らない。この違いを無視して係数が一致しないと騒ぐと、実装バグと定式化の違いを混同する。

### 4.2 双対と座標更新

双対を最小化形式で書くと、

$$
F(\alpha)=\frac12\alpha^T(Q+D)\alpha-\mathbf1^T\alpha,
\qquad 0\le\alpha_i\le U_i,
$$

$$
Q_{ij}=y_i y_j\tilde x_i^T\tilde x_j.
$$

hinge は $U_i=C_i,D_{ii}=0$。squared hinge は $U_i=\infty,D_{ii}=1/(2C_i)$ とする。$C_i=0$ は逆数を取らず、$\alpha_i=0$ に固定する。

現在の重みを保持していれば、勾配と更新は

$$
G_i=y_i\tilde w^T\tilde x_i-1+D_{ii}\alpha_i,
$$

$$
\alpha_i^{new}=
\operatorname{clip}\left(
\alpha_i-\frac{G_i}{\|\tilde x_i\|^2+D_{ii}},0,U_i
\right),
$$

$$
\tilde w\leftarrow\tilde w+
(\alpha_i^{new}-\alpha_i)y_i\tilde x_i
$$

で済む。dense では一座標あたり $O(d)$、CSR ではその行の非ゼロ要素に比例する。

この実装では局所 RNG による Fisher–Yates シャッフルを用い、複数モデルが共有するグローバル乱数状態を持たせない。座標順序は `random_state` で固定できる。

### 4.3 線形回帰の非滑らかさを消さない

`LinearSVR` は符号付き双対変数 $\beta$ を使う。双対の最大化目的関数は、

$$
D_{dual}=y^T\beta-\epsilon\|\beta\|_1
-\frac12\|\tilde w\|^2
-\frac12\sum_i D_{ii}\beta_i^2.
$$

$\epsilon|\beta_i|$ を含むため、単に微分して Newton step を打つのではなく、soft-thresholding を使う。

$$
S_\lambda(z)=\operatorname{sign}(z)\max(|z|-\lambda,0).
$$

通常損失では $|\beta_i|\le C_i$、二乗損失では無制約にし、対応する対角項を加える。ゼロ点の停止判定も単なる勾配ゼロではなく、劣勾配区間を考慮する。

## 5. カーネル SVM――三つの問題を共通 SMO に落とす

### 5.1 共通形

カーネル分類、回帰、one-class を次の問題に変換する。

$$
\min_\alpha \frac12\alpha^TQ\alpha+p^T\alpha,
\qquad 0\le\alpha_i\le C_i,
\qquad s^T\alpha=r,
$$

$$
Q_{ij}=s_i s_j K(x_{m_i},x_{m_j}),\qquad s_i\in\{-1,+1\}.
$$

$m_i$ は双対変数から元の標本への対応である。

| モデル | 双対変数 | 線形項 p | 等式制約 |
|---|---|---|---|
| C-SVC | n 個 | -1 | $y^T\alpha=0$ |
| epsilon-SVR | 2n 個の $(\alpha^+,\alpha^-)$ | $(\epsilon-y,\epsilon+y)$ | $\sum\alpha^+-\sum\alpha^-=0$ |
| one-class | n 個 | 0 | $\sum\alpha=1$ |

SVR のためにカーネル行を二倍保存する必要はない。二つの双対変数が同じ標本を参照するだけなので、キャッシュするのは元の $K$ の行でよい。

### 5.2 等式制約を壊さない二変数更新

二つの変数を、

$$
\Delta\alpha_i=s_i t,\qquad
\Delta\alpha_j=-s_j t
$$

と更新すれば、$s^T\alpha$ の変化が相殺される。

勾配 $G=Q\alpha+p$ と $v_i=-s_iG_i$ を使うと、この方向の曲率は

$$
\eta=K_{m_i m_i}+K_{m_j m_j}-2K_{m_i m_j}
$$

になる。実行可能な集合から最も KKT 違反の大きい $i$ を選び、$j$ は二次情報を使った指標

$$
\frac{(v_i-v_j)^2}{\max(\eta,10^{-12})}
$$

で選ぶ。最後に step を箱制約で許される範囲へクリップする。二次情報を用いる作業集合選択の参考は Fan・Chen・Lin の論文である。

ここで曲率を小さな正数へ置き換えても、不定値カーネルが PSD に変わるわけではない。sigmoid や任意の callable では、停留条件を満たすことと大域最適解を得ることを区別する必要がある。

### 5.3 LRU 行キャッシュ

毎回すべてのカーネルを再計算するのは遅い。一方、全 $n\times n$ Gram を必ず保持する設計は大きな問題で困る。

fastsvm は標本番号からキャッシュスロットへの表と双方向リストを持つ。検索、LRU 先頭への移動、追い出しは $O(1)$。データ本体は連続した float64 の行バッファに置く。

`cache_size` の単位は **MiB、二値モデル一つ当たり**。最低二行を保持するため、極端に小さな指定では二行分が設定量を超える。作業配列や複数クラスの同時学習もあるので、プロセス全体のメモリ上限を意味するわけではない。

callable、precomputed、全固有値による PSD 検査は、明示的な全 Gram を使う別経路である。「どの機能を有効にしても O(n) メモリ」という設計ではない。

### 5.4 shrinking は最終検査を省略する理由にならない

境界に固定されていて更新されそうにない変数を作業集合から外し、探索する変数を減らす。

ただし今回の実装では、外した変数の勾配も更新し続ける。作業集合だけで収束候補になったときには、全変数を復帰して KKT 残差を確認する。定期的な復帰も行う。

これは LIBSVM の内部と同一の aggressive shrinking / gradient reconstruction ではない。保守的な設計だが、古い勾配を収束の証拠として使わないことを優先した。

## 6. Cython で速くする場所、速くしない場所

### typed memoryview と nogil

密行列は `const double[:, ::1]`、ベクトルは `double[::1]`、CSR の添字は int64 で受け取る。形状・範囲・有限値を検査してから、Python のオブジェクト演算を含まないホットループへ入る。

内側関数には `noexcept nogil` を使う。`nogil` を付けるだけで単一スレッドが高速になるわけではない。役割は、native 演算中に GIL を保持せず、独立した仕事を並行実行できるようにすることだ。Cython 公式資料にもこの区別が説明されている。

学習時は共有する alpha / gradient を無造作に並列更新しない。多クラスの独立した二値学習を joblib のスレッドで実行し、予測や Gram 計算は標本行ごとの OpenMP 並列化を利用する。

小さな処理にまでスレッド起動コストを払わないよう、OpenMP は仕事量が閾値を超える場合だけ有効にする。OpenMP を使う経路は Python のメインスレッドからに限定し、ワーカースレッドから呼ばれた場合は逐次経路にする。

### RBF は差分を直接二乗する

$$
\|x-z\|^2=\|x\|^2+\|z\|^2-2x^Tz
$$

は恒等式だが、浮動小数点では両辺の安定性が同じとは限らない。$x$ と $z$ が共通の巨大オフセットを持つ場合、右辺は大きな量同士の差になる。

今回は $x_j-z_j$ を直接求めて二乗する。さらに四つの独立した累積器を使い、加算の依存鎖を短くした。ただし「この変更だけで何倍になった」という個別の因果効果は測定していない。示すのは完成した実装全体の比較である。

### 予測で巨大な中間行列を作らない

組み込みカーネルの予測は、

$$
f(x)=\sum_{i\in SV}\beta_iK(x,x_i)+b
$$

を行ごとに直接集計する。test × support vector の Gram 全体は作らない。linear カーネルでは学習後に重みベクトルへ畳み込み、行列積として予測する。

一方、入力 Hessian や固有値分解は必要なときだけ呼ぶ別機能に分ける。「解析機能が多いから、普通の予測でも毎回 Hessian を計算する」といった設計にはしない。

### 数値規約を壊す最適化は既定にしない

`-ffast-math` は使わない。NaN/Inf 検査、浮動小数点の扱い、再現性を雑にして速度を作らないためだ。入力を受理した後でも目的値や決定値が overflow すれば例外を返す。

`FASTSVM_DEBUG=1` では Cython の境界チェックと未最適化コンパイルを有効にできる。release の unchecked indexing を使う前提として、検証済みの配列だけを低レベル関数へ渡す。

## 7. 予測値だけでなく、解の状態を見る

### 7.1 主双対ギャップと KKT

小さな例を作る。

```python
import numpy as np
from fastsvm import SVC

rng = np.random.default_rng(19)
X_demo = rng.normal(size=(100, 3))
y_demo = (X_demo[:, 0] * X_demo[:, 1] > 0).astype(int)

model = SVC(C=3, gamma=0.4, tol=1e-8, history=True).fit(X_demo, y_demo)
report = model.optimization_report()[0]

for name in (
    "primal_objective", "dual_objective", "duality_gap",
    "relative_duality_gap", "kkt_violation",
    "equality_residual", "box_violation", "converged",
):
    print(name, report[name])

dual = model.dual_problem()
print("y^T alpha =", dual["signs"] @ dual["alpha"])
print("history shape =", model.optimization_history_[0].shape)
```

同梱のこの例を実行したところ、相対主双対ギャップは約 $4.25\times10^{-10}$、KKT 残差は約 $6.74\times10^{-9}$、等式制約残差は約 $3.11\times10^{-15}$ だった。これは当該データ・ビルドでの結果であり、全入力で保証する固定精度ではない。

主双対ギャップは

$$
\mathrm{gap}=P-D_{dual}
$$

で、最大化形式の双対目的値を引く。PSD と実行可能性の仮定が重要であり、不定値の問題で同じ数式を計算しただけでは大域最適性の証明にならない。

`max_iter` に達した場合も、最後の更新後の状態で全体残差を再計算する。未収束なら `ConvergenceWarning` を出し、`converged_=False` にする。「反復が終わった」と「収束した」は別の事実である。

### 7.2 入力勾配と Hessian

RBF について $\delta_i=x-x_i$ とすると、

$$
\nabla f(x)=-2\gamma\sum_i\beta_iK(x,x_i)\delta_i,
$$

$$
\nabla^2f(x)=\sum_i\beta_iK(x,x_i)
\left(4\gamma^2\delta_i\delta_i^T-2\gamma I\right).
$$

これを有限差分で近似するのではなく、解析式として実装した。

```python
gradient = model.decision_gradient(X_demo[:2])
hessian = model.decision_hessian(X_demo[:2])
print(gradient.shape)  # (2, 3)
print(hessian.shape)   # (2, 3, 3)
```

linear / polynomial / sigmoid にも対応する。Laplacian は折れ点があるので、`sign(0)=0` の勾配選択だけを返し、Hessian は提供しない。callable / precomputed から入力微分を勝手に推測することもしない。

多クラスでは `model=0` のように対象の二値モデルを指定する。微分の符号は `class_pairs_` の **二番目のクラスを正** とする内部スコアに対応する。sklearn 互換の公開 OvO スコアは一番目が正なので、比較時には符号を反転する。

また、これは **モデルへ入力した座標** に関する微分だ。標準化 $z_j=(x_j-\mu_j)/s_j$ を挟んだ場合、元の座標へ戻すには連鎖律を使う。

$$
\frac{\partial f}{\partial x_j}
=\frac1{s_j}\frac{\partial f}{\partial z_j},
\qquad
\frac{\partial^2f}{\partial x_j\partial x_k}
=\frac1{s_js_k}\frac{\partial^2f}{\partial z_j\partial z_k}.
$$

### 7.3 RKHS 距離を入力空間の距離と呼ばない

```python
print(model.rkhs_distance(X_demo[:2]))
```

返すのは $f(x)/\|w\|_{\mathcal H}$。特徴空間の超平面に対する符号付き距離である。元の入力空間で決定境界まで移動する最短距離ではない。ゼロノルムは例外にし、PSD を前提として解釈する。

## 8. カーネルを合成し、Gram のスペクトルを見る

```python
from fastsvm import (
    Kernel, pairwise_kernel, gram_diagnostics,
    project_psd, kernel_alignment, KernelCenterer,
)

k = Kernel("rbf", gamma=0.4) + 0.1 * Kernel("linear")
composite_model = SVC(kernel=k).fit(X_demo, y_demo)

K = pairwise_kernel(X_demo, gamma=0.4)
spectrum = gram_diagnostics(K)
print(spectrum["is_psd"], spectrum["rank"])
print(spectrum["min_eigenvalue"], spectrum["condition_number"])

centered = KernelCenterer().fit_transform(K)
print(kernel_alignment(K, K))
repaired = project_psd(K - 2 * np.eye(len(K)))
```

和・積・非負スカラー倍による PSD の閉性は、元のカーネルが PSD の場合に限る。sigmoid に線形カーネルを足したら自動で何でも安全になる、というものではない。

`gram_diagnostics()` は固有値、数値 rank、対称誤差、PSD 判定、条件数を返す。全固有値分解は $O(n^3)$ なので、通常の fit のたびに黙って実行しない。明示的に `check_psd=True` を指定した場合だけ、学習時にも確認する。

`project_psd()` は対称部分の固有値をクリップする射影だが、修復した学習 Gram から将来の test cross-Gram が自動的に決まるわけではない。数値的な行列修復と、全入力上で一貫したカーネルの定義は分けて考える。

precomputed を使う場合、予測入力の列数は support vector 数ではなく **学習標本数**。また `pairwise_kernel(..., gamma="scale")` は渡した X から値を推定するため、train / test の Gram を別々に作るときは共通の数値 gamma を使う。

## 9. 大きいデータでは近似写像を明示的に選ぶ

### Random Fourier Features

RBF 用に

$$
\omega\sim\mathcal N(0,2\gamma I),\quad
b\sim U(0,2\pi),\quad
z(x)=\sqrt{2/m}\cos(\Omega^Tx+b)
$$

を使う。

```python
from fastsvm import RBFSampler, LinearSVC

approximate = make_pipeline(
    StandardScaler(),
    RBFSampler(gamma=0.4, n_components=512, random_state=0),
    LinearSVC(C=0.5, random_state=0),
)
approximate.fit(X_train, y_train)
print(approximate.score(X_test, y_test))
```

固定次元の線形問題へ変換できるが、厳密な RBF SVM と同じ問題ではない。写像の次元による時間・メモリ・近似誤差の取引がある。

### Nyström

ランドマーク Gram の固有分解 $W=U\Lambda U^T$ から、

$$
Z=K(X,L)U_r\Lambda_r^{-1/2}
$$

を作る。

```python
from fastsvm import Nystroem

mapper = Nystroem(
    kernel="rbf", gamma=0.4, n_components=64,
    eigenvalue_tol=1e-12, random_state=0,
)
Z = mapper.fit_transform(X_demo)
print(Z.shape, mapper.n_features_out_)
```

小さい固有値を機械的に逆数にして数値を爆発させるのではなく、閾値以下を切り捨てる。そのため出力次元は `n_components` より小さくなることがある。顕著な負の固有値は拒否する。

### C を変えて観察する

```python
from fastsvm import regularization_path, margin_summary

path = regularization_path(
    LinearSVC(random_state=0), X_demo, y_demo,
    Cs=[0.01, 0.1, 1.0, 10.0], n_jobs=2,
)
for point in path:
    print(point["C"], point["training_score"], point["optimization"])

print(margin_summary(model, X_demo, y_demo)["quantiles"])
```

ここでいう path は **独立 fit による離散的な C 掃引** であり、warm start でも厳密な homotopy 法でもない。`training_score` は学習データ上のスコアなので、汎化性能の推定には別途 CV が必要になる。

## 10. sklearn 風でも、同じではない点

特に誤解しやすい違いを挙げる。

**確率校正。** `SVC(probability=True)` は、層化 CV と sigmoid 校正の ensemble を使う。分類器自体は fastsvm だが、校正の枠組みは `sklearn.calibration.CalibratedClassifierCV` に依存する。LIBSVM の pairwise coupling と同じではなく、追加学習も必要である。最大確率のクラスが通常の `predict()` と違う場合もある。

**OneClassSVM のスケール。** 双対質量を $\sum\alpha_i=1$ に正規化する。重みが一様なら、sklearn / LIBSVM の通常スコアを $\nu n$ で割ったスケールに対応する。境界の対応と、生スコアの一致は別である。境界上の標本では許容誤差・丸めだけでも予測符号が変わり得る。

**対応範囲。** exact kernel モデルは dense のみで、CSR は線形モデルや RFF の経路を使う。NuSVC / NuSVR、L1 正則化、primal/TRON、GPU、`partial_fit`、warm start は実装していない。`max_iter=-1` の無制限指定も受け付けない。

線形モデルの反復数は epoch、SMO は二変数更新回数である。同じ `n_iter_` という名前だけを見て直接比較しない。回帰器は `predict()` を提供し、分類用の `decision_function()` を無理に持たせない。

## 11. 検証――参照実装に近いだけで終わらせない

プロジェクト固有の pytest は **102 件**。次の三条件で全件通過した。

| 条件 | 結果 |
|---|---|
| native CPU 最適化 + OpenMP | 102 passed |
| 別ディレクトリでクリーンビルドした portable wheel のインストール後 | 102 passed |
| 境界チェック付き debug build | 102 passed |

同じ 102 ケースを三環境で実行したもので、306 個の異なる問題を検査したという意味ではない。

参照実装との予測・決定値比較に加え、小型問題は SciPy SLSQP と照合し、解析解を持つ二点問題も確認した。さらに $Q\alpha+p$ を独立に再構成し、ソルバーが更新し続けた gradient と突き合わせた。途中から壊れた gradient を使い、その同じ gradient だけで自分を検証する循環を避けるためだ。

重複点、ゼロ特徴、重みゼロ、CSR の重複要素、tiny LRU cache、shrinking on/off、OvO 係数再構成、有限差分による勾配・Hessian 検証、clone、pickle、Pipeline、precomputed CV、確率校正、OpenMP 予測の一致も含む。

別途、scikit-learn 1.8.0 の完全な estimator-check 一式を実行し、八つの推定器・変換器で **436 件、失敗ゼロ**だった。native と portable wheel の両方で確認している。

ただし、この検査では比較的厳しい tol と固定した数値 gamma を使用した。標本複製・重み変更の同値性検査で `gamma="scale"` の値まで変えると、同じ問題を比較したことにならないからだ。

また、LinearSVC / LinearSVR の検査中には、`tol=1e-10, max_iter=100000` で上限に達する `ConvergenceWarning` が記録された。チェックの失敗がないことを、すべての補助学習がその許容値まで収束したという意味にはしない。設定、警告、各チェック名は `docs/estimator_checks.json` に残してある。

実行済み OS は Linux、Python は 3.13.5。GitHub Actions には別 Python バージョン、macOS、Windows の設定も用意したが、設定ファイルの存在を実行実績とは呼ばない。

## 12. ベンチマーク――負けたケースも載せる

比較環境は Python 3.13.5、NumPy 2.3.5、SciPy 1.17.0、sklearn 1.8.0、Cython 3.2.4、GCC 14.2.0、Linux x86_64。

CPU の表示文字列は AMD EPYC 9V74 80-Core Processor だが、コンテナから見える論理 CPU は 5。測定は **一スレッド** に固定しており、80 コアを使用したという意味ではない。

両モデルを一回ずつウォームアップし、その後 9 回測定した中央値を使う。実行順は fastsvm→sklearn と sklearn→fastsvm を交互にする。

**倍率は sklearn の時間を fastsvm の時間で割った値。1 より大きい方が fastsvm に有利。**

| 課題 | 学習行数 × 特徴数 | fastsvm 学習 ms | sklearn 学習 ms | 学習倍率 | 推論倍率 |
|---|---:|---:|---:|---:|---:|
| 線形分類・dense | 3000 × 32 | 13.103 | 20.317 | 1.55× | 1.23× |
| 線形分類・CSR | 4500 × 1200 | 8.466 | 8.651 | 1.02× | 1.16× |
| RBF 二値分類 | 1800 × 16 | 17.242 | 18.602 | 1.08× | 3.50× |
| RBF 5クラス分類 | 1500 × 12 | 9.825 | 9.652 | 0.98× | 1.70× |
| RBF 回帰 | 1800 × 16 | 81.043 | 85.254 | 1.05× | 3.52× |
| 線形回帰 | 3000 × 32 | 10.020 | 9.189 | 0.92× | 0.96× |

データ生成・標準化は計測外。fit の入力検証、コピー、ソルバー、fastsvm の通常診断は計測に含めた。各ケースの `C`、gamma、損失、名目 tol 等を対応させ、線形参照実装は `dual=True` に固定した。

今回の分類予測は全ケースで sklearn と一致した。hold-out accuracy は dense 線形 0.843000、CSR 線形 0.867333、RBF 二値 0.980000、多クラス 0.986000。回帰 R² は RBF が 0.946987、線形が 0.982209 で、いずれも表示桁で参照実装と一致した。全六ケースで fastsvm は KKT 停止条件を満たし、測定対象 fit から未収束警告は出ていない。

ただし、同じ tol でも、二つの実装の停止規則まで同一になるわけではない。完全に同じ certified gap まで解かせた比較ではない。

さらに fastsvm はローカル CPU 向けビルド、参照は環境にインストール済みの sklearn バイナリである。参照側を同一コンパイラ・フラグで再ビルドした比較ではないので、コンパイラ最適化を含む実装全体の結果として読む必要がある。

1.02 倍や 0.98 倍は測定環境の揺れに近い可能性があり、有意差を主張しない。共有コンテナでの合成データ六ケースから、普遍的な速度ランキングは作れない。peak RSS と複数スレッドでの性能スケーリングも未測定である。

各回の時間、四分位点、設定、警告、決定値の最大絶対差、最適化診断は `benchmarks/results.json` に収録した。再測定は次で行う。

```bash
FASTSVM_NATIVE=1 FASTSVM_OPENMP=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python benchmarks/benchmark.py --repeat 9 --output benchmarks/my_results.json
```

環境変数を測定時に付けるだけで既存バイナリが再コンパイルされるわけではない。先に対応するフラグでビルドしておく。

## 13. 使い分けとまとめ

線形で十分なら `LinearSVC` / `LinearSVR` を使う。非線形性が必要でデータ規模が許すなら exact kernel。標本数が大きく、近似を許容できるなら RFF / Nyström と線形モデルを組み合わせる。

数値精度を調べたい場合は、accuracy だけではなく KKT、実行可能性、主双対ギャップを確認する。ただし PSD や損失・切片の定義を外したまま数値だけ眺めても、正しい保証にはならない。

高速化では、学習法の選択、配列配置、再利用する計算、作らない中間行列を先に決める。並列化やコンパイラフラグは、その上に重ねる。fastsvm は、それらを追跡可能な実装と検証コードにしたプロジェクトである。

コードは MIT ライセンス。学習ソルバーは独自実装だが、API・検証・校正のための scikit-learn 依存は残している。pickle / joblib で保存する場合は、信頼できるモデルファイルだけをロードすること。また support vector や学習由来の診断情報を含むため、配布時はその情報の扱いにも注意する。

## 参考資料

- Hsieh et al. (2008), [A Dual Coordinate Descent Method for Large-scale Linear SVM](https://www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf)
- Fan, Chen, Lin (2005), [Working Set Selection Using Second Order Information for Training Support Vector Machines](https://www.jmlr.org/papers/v6/fan05a.html)
- [LIBSVM project](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- [LIBLINEAR project](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
- [scikit-learn: Developing scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html)
- [scikit-learn: Kernel Approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html)
- [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Cython: Cython and the GIL](https://cython.readthedocs.io/en/latest/src/userguide/nogil.html)
- [Cython: Using Parallelism](https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html)

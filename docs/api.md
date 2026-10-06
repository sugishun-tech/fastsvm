# fastsvm 0.1.0 API リファレンス

Python 側は `BaseEstimator` と適切な mixin を継承し、`get_params()`、`set_params()`、`clone()`、`Pipeline`、`GridSearchCV` に対応する。学習はこのプロジェクトの Cython ソルバーで行う。scikit-learn の全引数・全メソッドの互換実装ではない。

## 入力と共通規約

`fit(X, y, sample_weight=None)` は自身を返す。`OneClassSVM.fit(X, y=None, sample_weight=None)` は教師ラベルを使用しない。分類ラベルは `classes_` に保存され、二値の正側は `classes_[1]`。

特徴入力の形状は `(n_samples, n_features)`、回帰の `y` は単一出力の一次元配列。内部計算は float64。有限値のみ受理し、NaN/Inf を含む入力は拒否する。float32 入力は float64 に変換する。データ検証・配列のコピーも実行時間とメモリ消費に含まれる。

| 項目 | 規約 |
|---|---|
| `C` | 正の有限実数。損失の **総和** に掛ける係数 |
| `sample_weight` | 非負で有限な一次元配列、またはスカラー。全ゼロは不可 |
| `class_weight` | 分類のみ。`None`、`"balanced"`、ラベル→非負重みの辞書 |
| ゼロ重み | 対応する双対変数をゼロに固定。分類では各クラスに正の有効重みが必要 |
| `tol` | 正の有限実数。ソルバー固有の KKT 残差に対する絶対許容値 |
| `max_iter` | 正の整数。`-1` による無制限指定は非対応 |
| `n_jobs` | joblib 規約の非ゼロ整数または `None`。`-1` は利用可能な全 CPU |
| `history` | 最適化履歴を保存するか。追加メモリ・記録コストがある |
| 未収束 | モデルを返し、`ConvergenceWarning`、`converged_=False`、`fit_status_=1` |
| 未学習での利用 | sklearn の `NotFittedError` |

`class_weight="balanced"` の SVC はクラスの標本数を基準にする。LinearSVC は `sample_weight` がある場合に重み付きクラス総量を基準にする。これは検証環境の sklearn 1.8.0 における両クラスの挙動に合わせたもの。どちらも最終的な標本コストは `C × sample_weight × class_weight` になる。

## 1. 線形モデル

```python
LinearSVC(
    *, C=1.0, loss="squared_hinge", tol=1e-4, max_iter=10000,
    fit_intercept=True, intercept_scaling=1.0,
    class_weight=None, random_state=None, shrinking=True,
    n_jobs=1, history=False,
)

LinearSVR(
    *, C=1.0, epsilon=0.1, loss="epsilon_insensitive", tol=1e-4,
    max_iter=10000, fit_intercept=True, intercept_scaling=1.0,
    random_state=None, shrinking=False, n_jobs=1, history=False,
)
```

入力は dense または SciPy CSR に対応する。他の SciPy sparse 形式は検証時に CSR へ変換される。重複要素の合算・添字の整列後、内部添字を int64 に統一する。利用者の sparse 入力を破壊的に正規化しない。

`LinearSVC.loss` は `"hinge"` / `"squared_hinge"`。L2 正則化のみ、双対座標降下法のみ。多クラスは OvR。`LinearSVR.loss` は `"epsilon_insensitive"` / `"squared_epsilon_insensitive"`、`epsilon >= 0`。

切片は `intercept_scaling=s` の人工特徴として学習され、正則化項 `b²/(2s²)` を持つ。`SVC(kernel="linear")` の非正則化切片とは異なる。`n_iter_` は coordinate-descent epoch 数で、複数モデルがある場合は最大値。

| 属性・メソッド | 内容 |
|---|---|
| `coef_` | 分類は `(1,d)` または `(n_classes,d)`。回帰は `(d,)` |
| `intercept_` | 二値・回帰は `(1,)`、多クラスは `(n_classes,)` |
| `dual_variables_` | `(n_binary_models,n_train)`。分類では alpha、回帰では符号付き beta |
| `n_iter_per_model_` | 各モデルの epoch 数 |
| `decision_function(X)` | 分類のみ。二値は一次元、多クラスはクラス別スコア |
| `predict(X)` | 分類ラベル、または回帰値 |
| `score(X,y)` | 分類 accuracy、回帰 R² |
| `optimization_report()` | モデルごとの主双対目的値・ギャップ・KKT 残差を返す |

`random_state` を固定すれば座標順序を再現できる。ただし異なる CPU・コンパイラ・BLAS を跨ぐ bit 単位の再現性は保証しない。`n_jobs` は多クラス分類の独立モデルに作用し、単一出力の LinearSVR の学習を並列化する引数ではない。

## 2. カーネルモデル

```python
SVC(
    *, C=1.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0,
    tol=1e-3, max_iter=1000000, cache_size=200.0,
    class_weight=None, decision_function_shape="ovr", break_ties=False,
    probability=False, calibration_cv=5, random_state=None,
    n_jobs=1, history=False, check_psd=False, shrinking=True,
)

SVR(
    *, C=1.0, epsilon=0.1, kernel="rbf", degree=3, gamma="scale", coef0=0.0,
    tol=1e-3, max_iter=1000000, cache_size=200.0,
    n_jobs=1, history=False, check_psd=False, shrinking=True,
)

OneClassSVM(
    *, nu=0.5, kernel="rbf", degree=3, gamma="scale", coef0=0.0,
    tol=1e-3, max_iter=1000000, cache_size=200.0,
    n_jobs=1, history=False, check_psd=False, shrinking=True,
)
```

exact kernel モデルは dense 入力のみ。疎行列に対する exact kernel 計算は実装しない。`nu` は `(0,1]`、`epsilon >= 0`、`degree` は非負整数。`gamma` は非負の数値、`"auto"`、`"scale"`。

`"auto"` は `1/d`。`"scale"` は学習特徴全体の分散を用いた `1/(d*X.var())`、分散ゼロなら 1。学習した値を推論でも使用する。重み付き分散ではない。明示的な Gram を渡す場合も、カーネルに関するハイパーパラメーターの妥当性は検証するが、既に与えられた Gram に gamma を再適用しない。

### カーネル

| `kernel` | 定義 |
|---|---|
| `"linear"` | `xᵀz` |
| `"rbf"` | `exp(-gamma * ||x-z||²)` |
| `"poly"` | `(gamma*xᵀz + coef0)**degree` |
| `"sigmoid"` | `tanh(gamma*xᵀz + coef0)` |
| `"laplacian"` | `exp(-gamma * ||x-z||₁)` |
| `"precomputed"` | 学習 `(n_train,n_train)`、予測 `(n_test,n_train)` |
| callable | `(X,Y) -> (len(X),len(Y))` の dense 配列を返す関数 |

precomputed の予測列は **全学習標本を学習時の順序で** 並べる。support vector 列だけの入力ではない。学習 Gram は正方・対称・有限でなければならない。`support_vectors_` は precomputed のとき `(0,0)`。

callable は学習時に全 Gram を構築するため O(n²) メモリ。推論時は support vector との cross-Gram を構築する。callable 内部の学習依存パラメーターを引数ごとに再推定しないこと。たとえば gamma を入力バッチごとに変更すると、一貫したカーネルにならない。

`cache_size` は二値モデル一つ当たりの MiB。最低二行を確保するため、非常に小さい設定では指定量を超える。OneClassSVM の n=1 は一行。LRU 行キャッシュとは別に dual / gradient / mapping 等の作業配列が必要で、設定値はプロセス全体のメモリ上限ではない。多クラス並列実行ではキャッシュも同時に複数確保される。

`check_psd=True` は **今回の学習 Gram** を全固有値分解で検査する。O(n³) 計算・O(n²) メモリであり、巨大データには適さない。全ての将来入力についてカーネルが PSD であることを証明する検査でもない。sigmoid、負の coef0 の polynomial、任意の callable では大域最適性を無条件に主張できない。

### 主な属性

`support_` は元の学習行番号、`support_vectors_` は対応する特徴、`n_support_` はクラス別個数。SVR / OneClassSVM は長さ 1 の個数配列。

`dual_coef_` は二値・回帰・one-class で `(1,n_support)`。多クラス SVC は sklearn 形式の `(n_classes-1,n_support)` に格納する。元の非負 alpha とは符号・配置が異なる。詳細検査には `dual_problem(model=k)` を使用する。

`intercept_` は二値問題ごとの切片。`coef_` は `kernel="linear"` の場合だけ存在する。`n_iter_` は各二値問題の SMO 二変数更新回数であり、線形モデルの epoch 数と比較できない。

### 多クラスの符号を明示する

SVC の学習は OvO、`class_pairs_` は `(class_i,class_j)` の辞書式組合せ順序。

- 二値 `decision_function(X)` は `classes_[1]` が正。
- 多クラス `decision_function_shape="ovo"` の各列は、sklearn に合わせて組の **一番目** のクラスが正。
- `pairwise_decision_function(X)` の各列は、二値ソルバーに合わせて常に組の **二番目** のクラスが正。

`decision_function_shape="ovr"` は pairwise の得票と有界な確信度補正をクラス別スコアに変換する。`break_ties=True` は ovr 指定の場合のみ有効。

微分と `rkhs_distance()` の多クラス列は **二番目が正** の内部二値モデルに対応する。公開 ovo スコアと比較する場合は符号を反転する。

### 確率校正

`probability=True` では、標本数条件を満たす層化 `calibration_cv` 分割で `CalibratedClassifierCV(method="sigmoid", ensemble=True)` を実行する。各 fold の分類器にも fastsvm を使う。LIBSVM の pairwise coupling を再実装したものではない。

`predict_proba(X)` / `predict_log_proba(X)` は probability 有効時のみ利用できる。校正モデル群と、通常の `predict()` が使う全データ学習済みモデルは別なので、最大確率のクラスと `predict()` が一致しない場合がある。追加 fit に相応の時間・メモリを要する。precomputed での内部校正は未対応。

### OneClassSVM の正規化

双対の質量を `sum(alpha)=1` に固定し、上限を `sample_weight_i/(nu*sum(sample_weight))` とする。重みが一様なら sklearn / LIBSVM の双対係数・生スコアを `nu*n_train` で割ったスケールに対応する。

`score_samples(X)` は切片なしの値、`offset_=-intercept_`、`decision_function=score_samples-offset_`。予測は非負が +1、負が -1。境界上では丸め誤差・停止許容値で符号が変わり得る。`fit_predict` に対応する。OneClassSVM は教師付き accuracy / R² を意味する `score` を提供しない。

## 3. 最適化と幾何の解析 API

全五推定器で `optimization_report()` がモデル別辞書のリストを返す。辞書はコピー。

| キー | 意味 |
|---|---|
| `primal_objective` | 実際の損失・正則化規約に従った主目的値 |
| `dual_objective` | **最大化形式** の双対目的値 |
| `duality_gap` | 主目的値から双対目的値を引いた値 |
| `relative_duality_gap` | gap / max(1,abs(primal)) |
| `kkt_violation` | 最終状態の全変数で再計算した停止残差 |
| `converged`, `n_iter` | 停止条件成立と反復数 |
| `equality_residual`, `box_violation` | カーネルモデルの制約違反 |
| `rkhs_norm_squared` | カーネルモデルの二乗ノルム。PSD を前提に解釈する |
| `weight_norm_squared` | 線形モデルの二乗ノルム。人工切片座標を含む |
| `cache_hits`, `cache_misses`, `cache_bytes` | カーネルモデルの行キャッシュ実績・確保量 |
| `min_active_size`, `shrinking_steps` | カーネル shrinking の動作状況 |

主双対目的値・gap・KKT は属性 `primal_objective_` / `dual_objective_` / `duality_gap_` / `kkt_violation_` にも配列で公開する。

`history=True` の `optimization_history_` はモデルごとの `(iteration, dual_objective, violation)` 配列。カーネルは主に 100 step ごと、線形は epoch ごと。記録用の途中残差は作業集合の走査時点の値であり、全体収束の証明ではない。最終行・最終 report を確認する。最終記録と周期記録が重なる場合に反復番号の重複があり得る。

カーネルモデルの `dual_problem(model=0)` は alpha、bounds、signs、mapping、gradient、training_indices、training_values のコピーを返す。Q 自体は大きいため保存・返却しない。線形モデルの双対変数は `dual_variables_` で参照する。

```python
model.decision_gradient(X_query, model=None)  # (n_query,d)
model.decision_hessian(X_query, model=None)   # (n_query,d,d)
model.rkhs_distance(X_query)                 # 二値・回帰: (n_query,)
```

この三つはカーネルモデルのみ。linear / RBF / polynomial / sigmoid の入力勾配・Hessian を実装する。Laplacian は `sign(0)=0` の勾配選択のみ。callable / precomputed の入力微分は未対応。

微分は固定された学習済み決定関数の入力微分であり、再学習を含む微分やハイパーパラメーター微分ではない。多クラスは `model=k` を明示する。Hessian は O(n_query*d²) の出力メモリを要する。解析処理は NumPy による別経路で、通常予測のホットループに含めない。

`rkhs_distance` は `f(x)/||w||_H`。入力空間での最近傍境界距離ではない。非正ノルムは拒否し、PSD カーネルであることは呼び出し側も確認する。

## 4. カーネル代数・スペクトル解析

```python
pairwise_kernel(X, Y=None, *, kernel="rbf", gamma="scale", degree=3,
                coef0=0.0, n_jobs=1)
gram_diagnostics(K, *, tolerance=1e-10)
project_psd(K, *, floor=0.0)
kernel_alignment(K, L, *, centered=True)
KernelCenterer()
```

`pairwise_kernel` は dense 行列を返すので、出力だけでも O(n*m) メモリが必要。`gamma="scale"` は引数 X から都度計算する。学習 Gram と test cross-Gram を別々に作る場合は **共通の数値 gamma** を渡す。

`gram_diagnostics` は対称誤差、PSD 判定、固有値列、数値 rank、条件数を返す。閾値は `tolerance * max(1,spectral_radius)`。`condition_number` は full-rank の正定値行列でのみ有限。`positive_spectrum_condition` は正の固有値部分だけの比であり、不定値行列全体の condition number ではない。

`project_psd` は正方行列の対称部分を、固有値が `floor >= 0` 以上の集合へ Frobenius ノルムで射影する。元のカーネルそのものを全入力上で修復する関数ではない。

`kernel_alignment` は規格化 Frobenius 内積。既定では双方を中心化し、片方がゼロノルムなら 0 を返す。`KernelCenterer.fit` は正方学習 Gram、`transform` は学習標本数を列数とする rectangular Gram を受け取る。

```python
from fastsvm import Kernel
k = 2 * Kernel("rbf", gamma=0.3) + Kernel("rbf", gamma=1.0) * Kernel("linear")
```

和・積・非負スカラー倍を callable として合成する。PSD の閉性は構成要素が PSD の場合に限る。`operation/left/right` は式木の内部表現であり、通常は直接指定しない。

## 5. 近似特徴写像

```python
RBFSampler(*, gamma=1.0, n_components=512, random_state=None)
Nystroem(*, kernel="rbf", gamma=1.0, degree=3, coef0=0.0,
         n_components=256, eigenvalue_tol=1e-12, random_state=None, n_jobs=1)
```

いずれも `fit` / `transform` / `fit_transform` / `get_feature_names_out`。RBFSampler は dense / CSR を受け取り、dense な乱数 Fourier 特徴を出力する。gamma は数値のみ。

Nystroem は dense 特徴入力、ランダムなランドマーク、PSD の数値 rank による切り捨てを使う。負の固有値が許容範囲を超える場合は拒否する。`n_features_out_` は有効 rank で、`n_components` より小さいことがある。precomputed は未対応。ランドマーク数が標本数を超えると警告して全標本を使用する。

## 6. その他の解析

```python
regularization_path(estimator, X, y, Cs, *, sample_weight=None, n_jobs=1)
margin_summary(estimator, X, y)
```

`regularization_path` は C ごとの独立 clone/fit のリストを返す。各要素は `C`, `model`, `fit_seconds`, `training_score`, `optimization`。入力順を保つ。warm start や厳密 homotopy ではない。`training_score` を交差検証結果と解釈しない。

`margin_summary` は二値分類専用。符号付き functional margin、分位点、最小値、実際の予測誤り率、非正 margin の比率、margin 内側の比率、平均 hinge loss を返す。margin=0 のタイブレークを区別するため、`error_fraction` と `nonpositive_margin_fraction` は別の値になることがある。

## 対応しない機能と保存時の注意

NuSVC / NuSVR、L1 正則化、primal/TRON、GPU、オンライン `partial_fit`、warm start、完全な sklearn metadata-routing 設定、全ての sklearn 固有オプションは未提供。対応外のコンストラクター引数を黙って無視しない。

pickle/joblib のラウンドトリップは検証済みだが、ロードは信頼できるファイルだけに限定する。モデルには support vector と診断用の双対変数・学習行番号・学習スコア等が残る。モデルファイルを共有するときは学習データ由来情報の扱いにも注意する。callable を含む場合は、その callable 自身が直列化可能である必要がある。

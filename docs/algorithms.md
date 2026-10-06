# fastsvm の数理と実装上の不変条件

本書の数式は、名前が似た別の最適化問題ではなく、実際の `fastsvm 0.1.1` が解く問題を示す。
添字 `i` は標本、`j` は特徴、`n` は標本数、`d` は特徴数とする。
最適化のホットループは `src/fastsvm/_core.pyx` にあり、C にコンパイルされる。

## 1. 線形分類：損失関数と切片の定義

ラベルは内部で `-1,+1` に符号化する。切片を使う場合、

$$\tilde x_i=(x_i,s),\qquad \tilde w=(w,b/s)$$

とする。`s=intercept_scaling` であり、`fit_intercept=False` の場合は切片の座標を持たない。

`LinearSVC(loss="hinge")` が最小化する主問題は

$$P=\frac12\|\tilde w\|^2+\sum_i C_i\max(0,1-y_i\tilde w^T\tilde x_i).$$

`squared_hinge` では損失を二乗する。`C_i = C × sample_weight_i × class_weight_i`。
ここは **損失の平均ではなく総和** である。標本数を変えたとき、同じ `C` が同じ「平均損失に対する正則化強度」を意味するわけではない。

切片の正則化項は `b²/(2s²)`。したがって、切片を正則化しない `SVC(kernel="linear")` とは原則として別問題である。

### 双対問題

$$\min_{0\le\alpha_i\le U_i}
F(\alpha)=\frac12\alpha^T(Q+D)\alpha-\mathbf1^T\alpha,
\quad Q_{ij}=y_iy_j\tilde x_i^T\tilde x_j.$$

hinge では `U_i=C_i, D_ii=0`、squared hinge では `U_i=∞, D_ii=1/(2C_i)`。
`C_i=0` は固定変数 `alpha_i=0` として扱い、逆数を計算しない。

各座標の更新は

$$G_i=y_i\tilde w^T\tilde x_i-1+D_{ii}\alpha_i,$$

$$\alpha_i^{new}=\operatorname{clip}
\left(\alpha_i-G_i/(\|\tilde x_i\|^2+D_{ii}),0,U_i\right).$$

続いて

$$\tilde w\leftarrow\tilde w+(\alpha_i^{new}-\alpha_i)y_i\tilde x_i$$

を行う。全 Gram 行列は不要である。dense では一巡 `O(nd)`、CSR では概ね `O(nnz+n)`。
ゼロ特徴・切片なし・hinge のゼロ曲率座標は、目的関数が減少する上限 `C_i` を選ぶ。

### 停止条件

投影勾配を

$$PG_i=\begin{cases}
\min(G_i,0)&\alpha_i=0\\
\max(G_i,0)&\alpha_i=U_i\\
G_i&0<\alpha_i<U_i
\end{cases}$$

と定め、最終状態で全座標を再走査し、`max_i |PG_i| <= tol` を確認する。
各 epoch 中の最大値だけを、そのまま最終解の証明として流用しない。

### 主双対ギャップ

最大化形式の双対目的関数は

$$D_{dual}=\sum_i\alpha_i-\frac12\|\tilde w\|^2-\frac12\sum_iD_{ii}\alpha_i^2.$$

`duality_gap_ = P - D_dual`。squared hinge の補正項は `alpha_i²/(4C_i)` になる。
丸めによりごく小さな負値になる可能性があるため、API では都合よくゼロに丸めず、計算値を公開する。

## 2. 線形回帰：非滑らかな項を正確に処理する

`LinearSVR` は、符号付き双対変数 `beta_i` を直接更新する。

$$D_{dual}=y^T\beta-\epsilon\|\beta\|_1
-\frac12\|\tilde w\|^2-\frac12\sum_iD_{ii}\beta_i^2,
\qquad\tilde w=\sum_i\beta_i\tilde x_i.$$

通常の epsilon-insensitive 損失では `|beta_i|<=C_i, D_ii=0`。
二乗損失では `beta_i` は無制約、`D_ii=1/(2C_i)` である。

$$G_i=\tilde w^T\tilde x_i-y_i+D_{ii}\beta_i,\qquad
q_i=\|\tilde x_i\|^2+D_{ii},$$

$$\beta_i^{new}=\operatorname{clip}
\left(S_{\epsilon/q_i}(\beta_i-G_i/q_i),-U_i,U_i\right),$$

$$S_\lambda(z)=\operatorname{sign}(z)\max(|z|-\lambda,0).$$

これは L1 項を無視した Newton 法ではなく、座標部分問題に対する soft-thresholding 更新である。
停止時には、ゼロ点の劣勾配区間と上下限の法線錐を考慮した KKT 残差を再計算する。

## 3. 非線形 SVM を共通の箱制約付き QP に落とす

C-SVC、epsilon-SVR、one-class SVM を次の形で解く。

$$\min_\alpha F(\alpha)=\frac12\alpha^TQ\alpha+p^T\alpha,$$

$$0\le\alpha_i\le C_i,\qquad s^T\alpha=r,$$

$$Q_{ij}=s_is_jK(x_{m_i},x_{m_j}),\qquad s_i\in\{-1,+1\}.$$

`m_i` は「双対変数から元の標本への対応」。epsilon-SVR では同じ標本が二度現れる。

| 問題 | 双対変数 | 符号 s | 線形項 p | 等式制約 |
|---|---|---|---|---|
| C-SVC | n 個の alpha | y | -1 | yᵀalpha=0 |
| epsilon-SVR | (alpha+, alpha-) の 2n 個 | (+1, -1) | (epsilon-y, epsilon+y) | sum(alpha+)-sum(alpha-)=0 |
| one-class | n 個の alpha | +1 | 0 | sum(alpha)=1 |

### SMO の二変数更新

実行可能な方向を

$$\Delta\alpha_i=s_it,\qquad\Delta\alpha_j=-s_jt$$

と置くと、等式制約は相殺される。勾配を `G=Q alpha+p`、`v_i=-s_i G_i` とすると、

$$\frac{dF}{dt}=-v_i+v_j,$$

$$\eta=K_{m_im_i}+K_{m_jm_j}-2K_{m_im_j}.$$

まず増加可能集合 `I_up` から `v_i` 最大の変数を選ぶ。次に減少可能集合 `I_low` の候補について

$$(v_i-v_j)^2/\max(\eta,10^{-12})$$

を比較する二次情報ベースの作業集合選択を行う。これは候補順位の近似指標であり、実際の改善量そのものではない。
実際の step は `(v_i-v_j)/eta` を両変数の実行可能な移動量でクリップする。
正でない曲率には小さな代替値を使い、箱制約の端点へ進める。

更新後、実際に生じた `delta alpha` を使って全勾配を更新する。
箱制約で制限された変数は端点に正確に設定する。これにより `alpha==0` / `alpha==C` による集合判定が端点付近の丸めで不安定になりにくい。

**正でない曲率を置き換えただけで、不定値カーネルの非凸問題が凸になるわけではない。**
その場合、KKT 残差が小さくても大域最適性の保証にはならない。

### KKT 違反量と切片

$$v_{max}=\max_{i\in I_{up}}v_i,\qquad
v_{min}=\min_{j\in I_{low}}v_j,$$

$$\operatorname{violation}=\max(0,v_{max}-v_{min}).$$

箱制約の内点にある変数では `b=-s_i G_i` が成り立つ。内点があればその平均を使い、なければ KKT 区間の中点を使う。
片側の区間しか定まらない退化例では有限な端点を採用する。

`max_iter` 終了時にも、最後の更新後の全変数で KKT 違反量を再計算する。
未収束を成功扱いせず、`ConvergenceWarning` と `fit_status_=1` を返す。

## 4. 共通ソルバーでも主目的関数は問題別に異なる

`f_i=sum_j beta_j K(x_i,x_j)+b`、`beta_j=sum_{m_k=j} s_k alpha_k` と置く。

C-SVC は

$$P=\frac12\|w\|_\mathcal H^2+\sum_iC_i\max(0,1-y_if_i).$$

epsilon-SVR は

$$P=\frac12\|w\|_\mathcal H^2+\sum_iC_i\max(0,|y_i-f_i|-\epsilon).$$

one-class は `b=-rho` として

$$P=\frac12\|w\|_\mathcal H^2-\rho+\sum_iC_i\max(0,\rho-\langle w,\phi(x_i)\rangle).$$

いずれも双対の最大化目的関数は `-F(alpha)` である。
PSD、実行可能性、各主問題の定式化を前提に `P-(-F)>=0` が成立する。

one-class では

$$C_i=\frac{u_i}{\nu\sum_j u_j},\quad\sum_i\alpha_i=1$$

とする。重みが一様なら、LIBSVM / sklearn の通常の双対係数・スコアは、この実装の値の `nu*n` 倍となる。
分類境界は対応するが、数値のスケールや停止許容値を無視して比較してはいけない。
学習点が境界上にある場合、丸めと許容誤差だけでも inlier/outlier の符号は変わり得る。

## 5. 高速化と不変条件を両立させる

線形ソルバーは局所 RNG による Fisher–Yates シャッフルを使う。グローバル C RNG は使わず、複数学習が乱数状態を奪い合わない。
CSR は Python 側で重複要素を合算し、ソートしてから渡す。添字は内部で 64 bit に統一する。

カーネルソルバーのキャッシュは、生の `K` の行を保持する。SVR の `2n` 双対変数に対しても、同じ標本の行を二重にキャッシュしない。
標本→スロット表と双方向リストで、検索・更新・追い出しは `O(1)`。
最少二行は保持するので、極小の `cache_size` 指定では二行分だけ指定値を超える。n=1 では一行で足りる。

カーネル shrinking は、境界上で現在の KKT 区間から十分離れた変数を作業集合から外す。
ただし全変数の勾配更新を継続し、候補収束時には全変数を復帰して再検査する。
LIBSVM と同じ aggressive shrinking / gradient reconstruction 実装ではない。
定期的な全復帰も行い、作業集合だけに依存した誤った収束判定を防ぐ。

RBF は `||x||²+||z||²-2xᵀz` ではなく差分を直接二乗する。
共通の巨大オフセットに対する桁落ちを避けるためである。四つの独立累積器で依存鎖を短くする。
これは浮動小数点の加算順序を変えるため、bit 単位の一致までは主張しない。

Cython の `nogil` と `noexcept nogil` を使い、内側ループに Python 呼び出しを入れない。
`nogil` 単独で単一スレッドが速くなるという説明は誤りであり、ここでは native ループ化と独立タスクの並行実行が役割を分担する。

## 6. 解析的微分

$$f(x)=\sum_i\beta_iK(x,x_i)+b.$$

RBF に対して `delta_i=x-x_i` とすると

$$\nabla f(x)=-2\gamma\sum_i\beta_iK(x,x_i)\delta_i,$$

$$\nabla^2f(x)=\sum_i\beta_iK(x,x_i)
\left(4\gamma^2\delta_i\delta_i^T-2\gamma I\right).$$

linear / polynomial / sigmoid についても解析式を実装する。
Laplacian は `sign(0)=0` を採用する勾配選択を返すが、折れ点で通常の微分が存在するという意味ではない。Hessian は未提供。
callable / precomputed は入力空間での微分を自動推測しない。

`rkhs_distance()` は `f(x)/||w||_H`。**入力空間における決定境界への最短距離ではない。**
PSD と正のノルムが前提。ゼロノルムでは例外にする。
多クラスの微分は OvO の「二番目のクラスが正」の各二値モデルを対象にし、`model` 指定を要求する。

## 7. 近似写像、スペクトル、正則化経路

`RBFSampler` は `omega~N(0,2gamma I)`、`b~Uniform(0,2pi)` により

$$z(x)=\sqrt{2/m}\cos(\Omega^Tx+b)$$

を作る。これはカーネル近似であり、有限個の特徴で厳密な RBF SVM と同じになるという主張はしない。

Nyström はランドマーク Gram 行列 `W=U Lambda Uᵀ` に対して

$$Z=K(X,L)U_r\Lambda_r^{-1/2}$$

を使う。小さな固有値を切り捨てるため、出力次元は `n_components` より小さくなり得る。
顕著な負の固有値は拒否する。固有値切り捨ては単なる実装上の丸めではなく、近似を定めるパラメーターである。

`gram_diagnostics()` の全固有値分解は `O(n³)`、Gram は `O(n²)` メモリ。
`project_psd()` は対称部分の固有値をクリップするが、修復済み学習 Gram から将来の cross-Gram が自動的に定義されるわけではない。
`KernelCenterer` はテスト行列を、学習時の列平均・全平均を使って中心化する。

`regularization_path()` は指定した `C` ごとの独立 fit である。
厳密な homotopy / piecewise-linear path 算法でも warm start でもない。返す score は学習データ上の値で、汎化性能の推定ではない。

## 参考資料

- Fan, Chen, Lin (2005), [Working Set Selection Using Second Order Information for Training Support Vector Machines](https://www.jmlr.org/papers/v6/fan05a.html)
- Hsieh et al. (2008), [A Dual Coordinate Descent Method for Large-scale Linear SVM](https://www.csie.ntu.edu.tw/~cjlin/papers/cddual.pdf)
- [LIBSVM project](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- [LIBLINEAR project](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
- [Cython: Cython and the GIL](https://cython.readthedocs.io/en/latest/src/userguide/nogil.html)
- [Cython: Using Parallelism](https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html)

上記は算法・API の参考資料であり、これらのソルバー実装コードの転載ではない。

# 実行済み検証結果

対象: fastsvm 0.1.0。記録日: 2026-10-06。
これは実行結果と限界の記録であり、形式検証・無欠陥の証明・全 OS 対応の保証ではない。

## 1. 実行環境

Linux x86_64、Python 3.13.5、NumPy 2.3.5、SciPy 1.17.0、scikit-learn 1.8.0、Cython 3.2.4、pytest 9.0.2、GCC 14.2.0。

## 2. プロジェクト固有 pytest

| ビルド条件 | 結果 |
|---|---|
| native CPU 最適化 + OpenMP、`OMP_NUM_THREADS=2` | **102 passed** |
| クリーンソースから portable wheel をビルド・別ディレクトリへインストール、OpenMP 無効 | **102 passed** |
| `FASTSVM_DEBUG=1`、Cython 境界チェック有効、最適化なし、OpenMP 無効 | **102 passed** |

同一の 102 ケースを三つのビルド条件で実行した。306 個の異なる数値ケースを用意したという意味ではない。pytest 実行の未処理警告はゼロ。意図した未収束・不定値カーネル等の警告はテスト内で明示的に検証する。

検証内容には、五推定器と sklearn の比較、独立な SciPy SLSQP による小型 QP の比較、解析的に解ける二点問題、full gradient の独立再構成、主双対ギャップ、等式制約・箱制約、shrinking on/off、二行 LRU キャッシュ、OvO 係数再構成を含む。

入力関連では dense / CSR、CSR 重複要素、読み取り専用配列、標本・クラス重み、ゼロ重み除外、重複点、全ゼロ特徴、単一標本の one-class、定数カーネル、巨大な共通オフセット、不正な Gram、overflow 拒否を検証した。

解析関連では入力勾配/Hessian の有限差分照合、Laplacian の非滑らかな点の扱い、Gram スペクトル、PSD 射影、kernel centering、alignment、RFF / Nyström、C 掃引、margin=0 の誤分類率と非正 margin 率の区別を検証した。

統合関連では Pipeline / GridSearchCV / precomputed CV / clone / pickle / 確率校正 / OpenMP 逐次・並列一致を検証した。さらに sklearn.svm の fit を例外に置き換えた状態でも、五つの fastsvm 学習器が動作することを検査した。確率校正は意図的に sklearn.calibration を使用する別機能である。

## 3. scikit-learn の全 estimator checks

| 推定器・変換器 | チェック数 | 失敗数 |
|---|---:|---:|
| SVC | 62 | 0 |
| SVR | 58 | 0 |
| OneClassSVM | 52 | 0 |
| LinearSVC | 63 | 0 |
| LinearSVR | 59 | 0 |
| RBFSampler | 47 | 0 |
| Nystroem | 47 | 0 |
| KernelCenterer | 48 | 0 |
| **合計** | **436** | **0** |

native/OpenMP ビルドと、別ディレクトリにインストールした portable wheel の双方で、この 436 チェックが通過した。debug ビルドでは固有 pytest のみを実施した。

検査に渡した設定は次のとおり。

```python
SVC(tol=1e-10, gamma=.4)
SVR(tol=1e-10, gamma=.4)
OneClassSVM(tol=1e-11, gamma=.4)
LinearSVC(tol=1e-10, max_iter=100000, random_state=0)
LinearSVR(tol=1e-10, max_iter=100000, random_state=0)
RBFSampler(n_components=20, random_state=0)
Nystroem(n_components=5, random_state=0)
KernelCenterer()
```

**デフォルト設定の全てが、どの sklearn バージョンのチェックでも通るという主張ではない。** とくに標本の複製や重み変更を使う同値性検査では、`gamma="scale"` の推定値自体が変わると同じ最適化問題にならないため、数値 gamma を固定した。

また、LinearSVC / LinearSVR の全体チェック中には、極端に厳しい tol=1e-10 に対して max_iter に達する `ConvergenceWarning` が記録された。チェックの失敗はゼロだが、全ての補助 fit がこの許容誤差まで収束したとは言っていない。警告を消して成功扱いしたのではなく、`docs/estimator_checks.json` の warnings に残している。

警告文:

```text
Dual coordinate descent reached max_iter before the KKT tolerance.
Standardize X, increase max_iter, or relax tol.
```

個々のチェック名・設定・status は `docs/estimator_checks.json`、portable wheel 側の集計は `docs/validation_summary.json`。

## 4. 例と配布確認

`examples/classification.py`、`mathematics.py`、`approximation.py`、`regression_and_outliers.py` の四つを実行し、いずれも正常終了した。Qiita 記事の Python コードブロック九つも掲載順に同じ名前空間で実行し、全て正常終了した。

portable wheel は `pip wheel --no-deps --no-build-isolation .` でクリーンな別ソースツリーから作成し、`pip install --no-deps --target ...` で独立した場所にインストールした。テスト時の import 元がそのインストール先であることと、`build_info()["openmp"] == False` を確認した。依存ライブラリは上記の実行済み環境を使っており、全依存関係を新規ダウンロードするテストではない。

PEP 517 build backend の `build_sdist` による source distribution の生成も実行し、ソルバー・API・記事・テストの収録を検査した。

GitHub 用 ZIP はソース配布であり、この検証用 wheel や生成された `_core.c`、CPU 依存 `.so` を含めない。GitHub リポジトリへの公開、PyPI 公開、GitHub Actions のリモート実行は行っていない。

## 5. 再実行

```bash
python -m pip install '.[dev]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/run_estimator_checks.py
```

CI 設定は Ubuntu の Python 3.10–3.13 / OpenMP 有無、および macOS・Windows の portable build を含む。これは今後の自動検証設定であり、今回の実行済み結果は Linux / Python 3.13.5 に限る。API の全互換、巨大問題の網羅、異なる BLAS・全 Python 対応範囲の実測、性能の普遍的優位は未検証。

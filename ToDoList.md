## To do

* **README.md 充実**

  * プロジェクト概要、使用方法、API 例、研究背景を追記。
    👉 PyPI ページは README.md がそのまま表示されるので、最小限の「インストール」「import 確認」「簡単なコード例」を必ず入れる。
  * PyPI 向けに最小限のインストール方法・サンプルコードを入れる。
  * バッジ（PyPI version / Python versions / License）を追加。

* **Secrets 登録（GitHub → Settings → Secrets → Actions）**

  * `TEST_PYPI_API_TOKEN`（TestPyPI の project-scoped token）
  * `PYPI_API_TOKEN`（PyPI の project-scoped token）
    ⚠️ Token の scope は **Project\:Publish** に限定。Account token は使わない。

* **ローカル検証**

  * `python -m build` → `python -m twine check dist/*` で配布物を検査。
  * sdist/whl に `LICENSE` / `NOTICE` が含まれることを確認。
  * ローカルで

    ```powershell
    pip install dist\chemomae-0.1.0-py3-none-any.whl
    python -c "import chemomae; print(chemomae.__version__)"
    ```

    → import テスト（assets 抜きで確認済み）。
    ⚠️ PowerShell の場合 `pip install dist/*.whl` は展開されないので、**ファイル名をフル指定**する。

* **TestPyPI 検証（自動）**

  ```bash
  git tag v0.1.0-rc1
  git push origin v0.1.0-rc1
  ```

  👉 Actions の `release-testpypi.yml` が走り、TestPyPI に公開される。
  👉 `pip install -i https://test.pypi.org/simple/ chemomae` でインストール確認。

* **本番リリース（自動）**

  ```bash
  git tag v0.1.0
  git push origin v0.1.0
  ```

  👉 Actions の `release-pypi.yml` が走り、本番 PyPI に公開される。
  👉 公開後は通常の `pip install chemomae` で利用可能。
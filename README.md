
# ranged_ray_torch_backend

RayとPyTorchの分散トレーニングをカスタムバックエンドで制御するPythonライブラリです。

## 特徴
- RayのWorkerGroupとPyTorchのプロセスグループを連携し、柔軟な分散初期化を実現
- 環境変数やTCP経由でマスターアドレス・ポートを自動管理
- 型ヒント・Protocolを活用した抽象化
- 独自のポート探索 (`_find_port`, `_is_free_port`) 実装

## 主要ファイル
- `ranged_ray_torch_backend/__init__.py`: 主要ロジック（分散初期化・プロトコル・設定クラス）
- `pyproject.toml`: 依存管理（`ray`, `torch`）

## 使い方
1. 必要な依存をインストール
	```bash
	pip install -r requirements.txt  # または pyproject.toml の poetry/pip 対応
	```
2. 分散トレーニングの初期化例:
	```python
	from ranged_ray_torch_backend import RangedTorchConfig, RangedTorchBackend
	# RayのWorkerGroupを用意し、RangedTorchConfigで初期化
	# 詳細は __init__.py を参照
	```

## 開発・テスト
- 標準的なPythonプロジェクト構成
- テストは `pytest` 推奨。`tests/` ディレクトリを作成し、
  ```bash
  pytest
  ```
  で実行

## 注意点
- Ray/PyTorchのバージョン互換性に注意
- 環境変数（`MASTER_ADDR`, `MASTER_PORT` など）の設定ミスに注意
- 分散環境のデバッグは各Workerのログを参照

## 参考: WorkerGroup初期化例
```python
master_addr, master_port = execute_single(0, get_address_and_port)
execute(set_env_vars, addr=master_addr, port=master_port)
```

---
詳細な設計・実装パターンは `.github/copilot-instructions.md` を参照してください。

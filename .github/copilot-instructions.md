# ranged_ray_torch_backend: Copilot Instructions

## 概要
このプロジェクトは、RayとPyTorchの分散トレーニングをカスタムバックエンドで制御するPythonライブラリです。主な機能は`ranged_ray_torch_backend/__init__.py`に実装されています。

## 主要コンポーネント
- `RangedTorchConfig`/`RangedTorchBackend`: RayのWorkerGroupとPyTorchのプロセスグループを連携。環境変数やTCPでマスターアドレス・ポートを管理。
- 独自のポート探索 (`_find_port`, `_is_free_port`) により、分散環境の初期化を柔軟に制御。
- `ExecuteSingle`, `Execute`, `ExecuteSingleAsync` プロトコルでWorkerへの関数実行を抽象化。

## データフロー・構造
- RayのWorkerGroupを使い、各WorkerでPyTorchのプロセスグループを初期化。
- マスターアドレス・ポートは環境変数またはTCPで決定。
- 各Workerは`execute_single_async`で非同期にセットアップされ、`ray.get`で同期。

## 開発・ビルド・テスト
- 標準的なPythonプロジェクト構成（`pyproject.toml`あり）。
- 依存: `ray`, `torch`（`pyproject.toml`で管理）。
- テストやビルドコマンドはREADME未記載。pytest等を利用する場合は`tests/`ディレクトリを作成し、`pytest`コマンドを推奨。

## コーディング規約・パターン
- 型ヒント・Protocolを積極利用。
- Ray/PyTorchの内部API（private usage）も利用しているため、pyright等の型チェックは`# pyright: ignore`で抑制。
- 環境変数による挙動制御（`MASTER_ADDR`, `MASTER_PORT`, `MASTER_MIN_PORT`, `MASTER_MAX_PORT`）。
- 例外処理は`RuntimeError`で統一。

## 参考ファイル
- `ranged_ray_torch_backend/__init__.py`: 主要ロジック・パターンの全てがここに集約。
- `README.md`: プロジェクト名のみ記載。詳細はコード参照。

## 例: WorkerGroupの初期化
```python
master_addr, master_port = execute_single(0, get_address_and_port)
execute(set_env_vars, addr=master_addr, port=master_port)
```

## 注意点
- Ray/PyTorchのバージョン互換性に注意。
- 環境変数の設定ミスは初期化失敗の原因。
- 分散環境のデバッグは各Workerのログを参照。

---
この内容で不明点や追加したい情報があればご指摘ください。
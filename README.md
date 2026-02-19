# NovelCharacterAI
[Novel2DialCorpus](https://github.com/ganbon/Novel2DialCorpus)で構築したコーパスを学習してキャラクターAIを構築するコード

### Python Enviroments
pythonの仮想環境ツールとして[uv](https://docs.astral.sh/uv/)が必要です
- uvのインストール
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- 仮想環境の構築
```shell
$ ./scripts/create_venv.sh
```

## Usage
### Create Train Dataset
- parameter
 - novel_path:小説対話コーパス
 - character_path:人物名リスト
 - output_path:出力先のpath
```shell
$ source .venv/bin/activate
$ python ./scripts/create_chracter_sample_utterance.py novel_path character_path output_path 
```

### Training
- parameter
 - dataset_path:学習データ
 - smaple_utterance_path:口調例データ
 - save_dir:モデル保存
 - model_name:事前学習モデル
```shell
$ source .venv/bin/activate
$ python ./scripts/train.py dataset_path sample_utterance_path save_dir model_name
```

### Dialogue Generate
- parameter
 - smaple_utterance_path:口調例データ
 - model_path:対話モデル
 - chracter_id:対話相手のキャラクターID
```shell
$ source .venv/bin/activate
$ python ./scripts/train.py sample_utterance_path model_name character_id
```

## Citation
```bibtex
@inproceedings{iwamoto-etal-2026-novel2dialcorpus,
    title = " Novel2DialCorpus:小説を用いた対話コーパスの自動構築手法",
    author = "岩本 和真 and 安藤 一秋",
    booktitle = "言語処理学会第32回年次大会発表論文集",
    year = "2026"
}
```
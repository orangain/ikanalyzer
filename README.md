# ikanalyzer

ikanalyzerはゲームソフト「スプラトゥーン3」のプレイ動画を分析するためのツールです。
現時点では、開発者個人の環境（macOS, GV-HDREC）向けに特化して開発されており、多様な環境での動作は保証されていません。

開発者はスプラトゥーン3や任天堂とは無関係の個人であり、任天堂からの公式なサポートはありません。
本ツールについて、任天堂やスプラトゥーン3の開発者に問い合わせることはお控えください。

本ツールは利用者本人のプレイ動画を分析することによって上達の一助になることを目的としています。他人への批判や攻撃、またはチートなど不正な目的で使用することはお控えください。

## Caution / 注意事項

本ツールは分析処理の一部でOpenAIのAPIを使用します。OpenAIのAPIの使用は必須ではありませんが、使用する場合には費用が発生し、動画の一部がOpenAIに送信されます。

OpenAIのAPIを使用しない場合、以下の制約があります。

- プレイヤー名のOCR精度が低下します。
- デスした際の相手プレイヤーを特定できません。

ライセンスファイルに記載の通り、本ツールを使用したことや使用できなかったことによる費用や損害について、開発者は一切の責任を負いません。利用者自身の責任においてご利用ください。

<details>
<summary>参考: OpenAI APIの費用の例</summary>
作者が2025/05/04に27試合分のイベントフレームを分析した際には、gpt-4o-miniで約256万トークンを消費し、約$0.4の費用がかかりました。トークン数は試合数や試合の内容によって異なります。これはあくまで参考値であり、実際にかかる費用を保証するものではありません。
</details>

## Pre-requisites / 必要なもの

- Git
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://www.ffmpeg.org)
- [tesseract](https://github.com/tesseract-ocr/tesseract)
- (Optional) [OpenAI API Key](https://openai.com/api/) (OpenAI APIを使用する場合のみ)

macOSの場合、Homebrewを使用して以下のようにインストールできます。その他の環境での動作は未確認です。

```
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install Python 3.10+
uv python install 3.13
# Install ffmpeg
brew install ffmpeg
# Install tesseract
brew install tesseract tesseract-lang
```

## Installation / インストール

```
git clone https://github.com/orangain/ikanalyzer
cd ikanalyzer
uv install
```

## Usage / 使い方

以下の手順で動画を分析します。

1. 動画を準備する
2. 連続録画した動画を試合ごとに分割する
3. 分割した動画からイベントフレームを抜き出す
4. OpenAIのAPIキーを設定する (Optional)
5. イベントフレームを分析する
6. 分析結果を出力する

### 1. 動画を準備する

キャプチャーボードなどを使ってプレイ動画を撮影します。保存場所の指定はありませんが、以降では `workspace/original_videos` ディレクトリに保存したものとして説明します。

1時間など、複数の試合を連続録画したものでOKです。現状では以下の条件で撮影された動画に対応しています。

- ゲームの言語: 日本語
- ルール: ナワバリバトル、ガチエリア、ガチヤグラ、ガチホコバトル、ガチアサリ
- 解像度: 1920x1080 (1080p)
- フレームレート: 3 fps以上 (29.97fps以外での動作は未確認)

**注意**: ゲームプレイの際、試合終了時にリザルト画面が自動で表示されない場合（連戦終了時やスケジュールの変更時）は、次の試合までにリザルト画面を手動で表示するようにしてください。

ロビーでLボタンを押してメニューを開き、Rボタンを押すとリザルト画面を表示できます。リザルト画面が録画に含まれていない場合、その試合を分析することはできません。

### 2. 連続録画した動画を試合ごとに分割する

動画を試合ごとに分割するために、以下のコマンドを実行します。

```bash
uv run split_video.py INPUT_VIDEO...
```

`INPUT_VIDEO`には、連続録画した動画のパスを指定します。次のように複数の動画のパスを指定することもできます。

```
uv run split_video.py workspace/original_videos/*.mp4
```

分割された動画は、 `workspace/videos` ディレクトリに保存されます。動画ファイル名は `{元の動画のファイル名}_{開始フレーム番号}.{元の動画の拡張子}` という形式になります。
例えば、`original.mp4` という動画を分割した場合、分割された動画は `videos/original_001422.mp4`, `videos/original_006804.mp4`, ... といった名前になります。

<details>
<summary>正常に動作した際に表示されるログの例</summary>

```
2025-05-06T19:51:59 [66336][INFO] Video loaded: workspace/original_videos/IOHD0020.MP4, fps: 29.97002997002997
2025-05-06T19:51:59 [66336][INFO] Processed 0 frames...
2025-05-06T19:52:01 [66336][INFO] Processed 900 frames...
2025-05-06T19:52:03 [66336][INFO] Processed 1800 frames...
2025-05-06T19:52:05 [66336][INFO] Processed 2700 frames...
2025-05-06T19:52:07 [66336][INFO] Processed 3600 frames...
2025-05-06T19:52:09 [66336][INFO] Processed 4500 frames...
2025-05-06T19:52:11 [66336][INFO] Processed 5400 frames...
2025-05-06T19:52:13 [66336][INFO] Processed 6300 frames...
2025-05-06T19:52:15 [66336][INFO] Processed 7200 frames...
2025-05-06T19:52:15 [66336][INFO] Frame matched. matched_frame_type: opening, frame_number: 7389
2025-05-06T19:52:17 [66336][INFO] Processed 8100 frames...
2025-05-06T19:52:19 [66336][INFO] Processed 9000 frames...
2025-05-06T19:52:21 [66336][INFO] Processed 9900 frames...
2025-05-06T19:52:23 [66336][INFO] Processed 10800 frames...
2025-05-06T19:52:25 [66336][INFO] Processed 11700 frames...
2025-05-06T19:52:27 [66336][INFO] Processed 12600 frames...
2025-05-06T19:52:29 [66336][INFO] Processed 13500 frames...
2025-05-06T19:52:31 [66336][INFO] Processed 14400 frames...
2025-05-06T19:52:33 [66336][INFO] Processed 15300 frames...
2025-05-06T19:52:35 [66336][INFO] Processed 16200 frames...
2025-05-06T19:52:37 [66336][INFO] Processed 17100 frames...
2025-05-06T19:52:39 [66336][INFO] Processed 18000 frames...
2025-05-06T19:52:39 [66336][INFO] Frame matched. matched_frame_type: result, frame_number: 18090
2025-05-06T19:52:39 [66336][INFO] ffmpeg -i workspace/original_videos/IOHD0020.MP4 -ss 0:04:05.546300 -to 0:10:04.603000 -c copy workspace/videos/IOHD0020_007389.MP4
ffmpeg version 7.1.1 Copyright (c) 2000-2025 the FFmpeg developers
  built with Apple clang version 16.0.0 (clang-1600.0.26.6)
  configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/7.1.1_2 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags='-Wl,-ld_classic' --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libaribb24 --enable-libbluray --enable-libdav1d --enable-libharfbuzz --enable-libjxl --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libssh --enable-libsvtav1 --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-audiotoolbox --enable-neon
  libavutil      59. 39.100 / 59. 39.100
  libavcodec     61. 19.101 / 61. 19.101
  libavformat    61.  7.100 / 61.  7.100
  libavdevice    61.  3.100 / 61.  3.100
  libavfilter    10.  4.100 / 10.  4.100
  libswscale      8.  3.100 /  8.  3.100
  libswresample   5.  3.100 /  5.  3.100
  libpostproc    58.  3.100 / 58.  3.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'workspace/original_videos/IOHD0020.MP4':
  Metadata:
    major_brand     : mp42
    minor_version   : 1
    compatible_brands: mp42avc1
    creation_time   : 2025-05-06T13:22:31.000000Z
    original_format : Video Capture
    original_format-eng: Video Capture
    comment         : I-O DATA GV-HDREC
    comment-eng     : I-O DATA GV-HDREC
  Duration: 04:31:26.27, start: 0.000000, bitrate: 9292 kb/s
  Stream #0:0[0x1](eng): Video: h264 (Constrained Baseline) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1920x1080 [SAR 1:1 DAR 16:9], 9037 kb/s, 29.97 fps, 29.97 tbr, 30k tbn (default)
      Metadata:
        creation_time   : 2025-05-06T13:22:31.000000Z
        vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 253 kb/s (default)
      Metadata:
        creation_time   : 2025-05-06T13:22:31.000000Z
        vendor_id       : [0][0][0][0]
Stream mapping:
  Stream #0:0 -> #0:0 (copy)
  Stream #0:1 -> #0:1 (copy)
Output #0, mp4, to 'workspace/videos/IOHD0020_007389.MP4':
  Metadata:
    major_brand     : mp42
    minor_version   : 1
    compatible_brands: mp42avc1
    comment-eng     : I-O DATA GV-HDREC
    original_format : Video Capture
    original_format-eng: Video Capture
    comment         : I-O DATA GV-HDREC
    encoder         : Lavf61.7.100
  Stream #0:0(eng): Video: h264 (Constrained Baseline) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1920x1080 [SAR 1:1 DAR 16:9], q=2-31, 9037 kb/s, 29.97 fps, 29.97 tbr, 30k tbn (default)
      Metadata:
        creation_time   : 2025-05-06T13:22:31.000000Z
        vendor_id       : [0][0][0][0]
  Stream #0:1(eng): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 253 kb/s (default)
      Metadata:
        creation_time   : 2025-05-06T13:22:31.000000Z
        vendor_id       : [0][0][0][0]
Press [q] to stop, [?] for help
[out#0/mp4 @ 0x60000230c000] video:714462KiB audio:11095KiB subtitle:0KiB other streams:0KiB global headers:0KiB muxing overhead: 0.032862%
frame=10740 fps=0.0 q=-1.0 Lsize=  725795KiB time=00:10:04.70 bitrate=9832.4kbits/s speed=2.15e+03x    
2025-05-06T19:52:39 [66336][INFO] Movie written: workspace/videos/IOHD0020_007389.MP4
...
2025-05-06T19:54:28 [66336][INFO] Saved 5 videos.
```

</details>

### 3. 分割した動画からイベントフレームを抜き出す

分割した動画からイベントフレーム（試合開始、キル、デス、表彰、リザルト）を抜き出すために、以下のコマンドを実行します。

```bash
uv run extract_event_frames.py INPUT_VIDEO...
```

`INPUT_VIDEO`には、分割した動画のパスを指定します。次のように複数の動画のパスを指定することもできます。

```
uv run extract_event_frames.py workspace/videos/*.mp4
```

分割した動画から抜き出されたイベントフレームは、 `workspace/event_frames/{元の動画のファイル名}` ディレクトリに保存されます。ファイル名は `frame_{フレーム番号}_{イベント名}.png` という形式になります。

同じディレクトリに生成される `metadata.json` というファイルは後の分析で使用します。削除しないでください。

<details>
<summary>正常に動作した際に表示されるログの例</summary>

```
2025-05-06T20:20:16 [66708][INFO] Using up to 5 processes.
2025-05-06T20:20:16 [66710][INFO] Video loaded: workspace/videos/IOHD0018_000486.MP4, fps: 29.97002997002997
2025-05-06T20:20:16 [66714][INFO] Video loaded: workspace/videos/IOHD0018_011934.MP4, fps: 29.97002997002997
2025-05-06T20:20:16 [66712][INFO] Video loaded: workspace/videos/IOHD0018_023292.MP4, fps: 29.97002997002997
2025-05-06T20:20:16 [66713][INFO] Video loaded: workspace/videos/IOHD0018_047016.MP4, fps: 29.97002997002997
2025-05-06T20:20:16 [66711][INFO] Video loaded: workspace/videos/IOHD0018_035082.MP4, fps: 29.97002997002997
2025-05-06T20:20:16 [66710][INFO] Processed 0 frames...
2025-05-06T20:20:16 [66714][INFO] Processed 0 frames...
2025-05-06T20:20:16 [66712][INFO] Processed 0 frames...
2025-05-06T20:20:16 [66711][INFO] Processed 0 frames...
2025-05-06T20:20:16 [66713][INFO] Processed 0 frames...
2025-05-06T20:20:16 [66711][INFO] Frame matched. matched_frame_type: opening, frame_number: 9
2025-05-06T20:20:16 [66710][INFO] Frame matched. matched_frame_type: opening, frame_number: 54
2025-05-06T20:20:17 [66713][INFO] Frame matched. matched_frame_type: opening, frame_number: 54
2025-05-06T20:20:17 [66712][INFO] Frame matched. matched_frame_type: opening, frame_number: 72
2025-05-06T20:20:17 [66714][INFO] Frame matched. matched_frame_type: opening, frame_number: 81
2025-05-06T20:20:20 [66714][INFO] Frame matched. matched_frame_type: kill, frame_number: 855
...
2025-05-06T20:20:45 [66710][INFO] Processed 5400 frames...
2025-05-06T20:20:45 [66714][INFO] Processed 5400 frames...
2025-05-06T20:20:46 [66713][INFO] Processed 5400 frames...
2025-05-06T20:20:46 [66712][INFO] Processed 5400 frames...
2025-05-06T20:20:46 [66711][INFO] Processed 5400 frames...
2025-05-06T20:20:46 [66714][INFO] Frame matched. matched_frame_type: death, frame_number: 5463
2025-05-06T20:20:47 [66712][INFO] Frame matched. matched_frame_type: kill, frame_number: 5553
2025-05-06T20:20:47 [66711][INFO] Frame matched. matched_frame_type: kill, frame_number: 5643
2025-05-06T20:20:48 [66713][INFO] Frame matched. matched_frame_type: award, frame_number: 5922
2025-05-06T20:20:49 [66711][INFO] Frame matched. matched_frame_type: death, frame_number: 5940
2025-05-06T20:20:49 [66712][INFO] Frame matched. matched_frame_type: death, frame_number: 6003
2025-05-06T20:20:50 [66713][INFO] Processed 6300 frames...
2025-05-06T20:20:50 [66713][INFO] Frame matched. matched_frame_type: result_lobby, frame_number: 6327
2025-05-06T20:20:50 [66710][INFO] Processed 6300 frames...
2025-05-06T20:20:50 [66713][INFO] Saved 14 event frames.
...
```

</details>

### 4. OpenAIのAPIキーを設定する (Optional)

分析でOpenAIのAPIを使用する場合は、 `.env` ファイルを作成し、以下のようにAPIキーを設定します。先述の通り、API利用によって発生した費用は利用者の負担となりますので、十分に注意してください。

```
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXX
```

OpenAIのAPIを使用しない場合は、 `.env` ファイルは不要です。

### 5. イベントフレームを分析する

分割した動画から抜き出したイベントフレームを分析するために、以下のコマンドを実行します。

```bash
uv run --env-file .env analyze_event_frames.py INPUT_DIR...
```

OpenAIのAPIを使用しない場合には `--env-file .env` オプションを省略してください。

`INPUT_DIR`には、イベントフレームが保存されたディレクトリのパスを指定します。次のように複数のディレクトリのパスを指定することもできます。

```
uv run --env-file .env analyze_event_frames.py workspace/event_frames/*
```

分析結果は同じディレクトリ内に `result.json` という名前で保存されます。
また、 `intermediate` ディレクトリに、分析の途中で生成されたデバッグ用のファイルやOpenAI APIのキャッシュが保存されます。
再実行時にキャッシュがヒットする場合にはOpenAI APIの呼び出しがスキップされるため、削除しないことをお勧めします。

`result.json` には、以下のような情報が含まれています。

* `version`: result.jsonのバージョン
* `id`: 分析対象の動画のID
* `team_result`: チームの勝敗
* `local_player`: 自分のプレイヤー情報
* `ally_players`: 味方プレイヤーの情報
* `enemy_players`: 敵プレイヤーの情報
* `events`: イベントの情報

### 6. 分析結果を出力する

分析結果を出力するために、以下のコマンドを実行します。

```bash
uv run export_results.py RESULT_FILE... -o OUTPUT_DIR
```

`RESULT_FILE`には、分析結果を含むディレクトリまたは `result.json` のパスを指定します。次のように複数のパスを指定することもできます。

```
uv run export_results.py workspace/event_frames/* -o workspce/results
```

`OUTPUT_DIR`には、出力先のディレクトリを指定します。指定したディレクトリが存在しない場合は自動で作成されます。

出力されるファイルは以下の通りです。

* `games.tsv`: 試合の情報を含むTSVファイル
* `players.tsv`: プレイヤーの情報を含むTSVファイル
* `events.tsv`: イベントの情報を含むTSVファイル
* `enemy_weapon_stats.tsv`: 相手のブキごとの統計情報を含むTSVファイル


## その他のコマンド

以下のコマンドは、主にデバッグや開発用のコマンドです。通常は使用しません。

* `dump_frames.py`: 動画からフレームを画像として抜き出す。

# ============================================================
# Stage 1: Build kytea from source (needs C++ toolchain)
# ============================================================
FROM docker.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS kytea-builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://www.phontron.com/kytea/download/kytea-0.4.7.tar.gz \
    && tar xzf kytea-0.4.7.tar.gz \
    && cd kytea-0.4.7 \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install

# ============================================================
# Stage 2: Runtime (devel → runtime で ~3GB 削減)
# ============================================================
FROM docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PROJECT_DIR=/root/ocr_cli \
    LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8 \
    TZ=Asia/Tokyo

# kytea: バイナリ + 共有ライブラリ + モデルデータのみ (ヘッダ不要)
COPY --from=kytea-builder /usr/local/bin/kytea /usr/local/bin/
COPY --from=kytea-builder /usr/local/lib/libkytea* /usr/local/lib/
COPY --from=kytea-builder /usr/local/share/kytea/ /usr/local/share/kytea/

# システムパッケージ: 1回の apt-get で完結、キャッシュ即削除
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        locales libgl1 libglib2.0-0 python3-pip \
    && locale-gen ja_JP.UTF-8 \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && ldconfig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 依存 (変更頻度の低い順 → レイヤーキャッシュ最適化) ----

# PyTorch (プレビルド wheel, ~2.5GB, 最も安定)
RUN pip3 install --no-cache-dir \
    torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 torchtext==0.16.1 \
    --index-url https://download.pytorch.org/whl/cu121

# メイン requirements (requirements.txt だけ先に COPY → コード変更で再実行されない)
COPY requirements.txt ${PROJECT_DIR}/requirements.txt
RUN pip3 install --no-cache-dir -r ${PROJECT_DIR}/requirements.txt

# mmcv (プレビルド wheel) → mmdet の順で入れる (mmdet が mmcv に依存)
RUN pip3 install --no-cache-dir \
    mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html \
    && pip3 install --no-cache-dir mmdet==3.3.0

# サブモジュール + 固有の依存
COPY submodules/ ${PROJECT_DIR}/submodules/
RUN pip3 install --no-cache-dir \
    -r ${PROJECT_DIR}/submodules/ruby_prediction/requirements.txt

# ---- アプリケーションコード (最も頻繁に変わる → 最後) ----
COPY cli/ ${PROJECT_DIR}/cli/
COPY main.py config.yml eval_config.yml README.md ${PROJECT_DIR}/

WORKDIR ${PROJECT_DIR}

# -*- coding: utf-8 -*-
"""
download_data.py —— 下载基底图库（无需 Kaggle 账号 / API token）。

数据源：Kaggle 数据集 `jenlooper/cheese`（All The Cheese，3234 张，6 个表皮工艺类目）。
Kaggle 的公共下载端点允许匿名访问：
    https://www.kaggle.com/api/v1/datasets/download/jenlooper/cheese

下载到  data/raw/cheese_kaggle.zip。prepare_dataset.py 直接从 zip 里读图，
不需要解压。

用法：
  python code/download_data.py
  python code/download_data.py --force        # 已有文件也重下
"""

import os, sys, time, argparse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

URL = "https://www.kaggle.com/api/v1/datasets/download/jenlooper/cheese"
DST = os.path.join(paths.RAW, "cheese_kaggle.zip")
EXPECT = 123_000_000          # 约 123 MB，仅做完整性粗校验


def size_of(p):
    return os.path.getsize(p) if os.path.exists(p) else 0


def download(force=False):
    cur = size_of(DST)
    if cur > 100_000_000 and not force:
        print(f"已存在 {DST}  ({cur/1048576:.1f} MB)，跳过。用 --force 强制重下。")
        return
    tmp = DST + ".part"
    start = size_of(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(URL, headers={
        "User-Agent": "Mozilla/5.0 (cheese-mold-detection dataset fetcher)",
        **({"Range": f"bytes={start}-"} if start else {}),
    })
    print(f"GET {URL}  (已缓存 {start/1048576:.1f} MB)")
    t0, last = time.time(), start
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab" if start else "wb") as f:
        total = int(r.headers.get("Content-Length") or 0) + start
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            last += len(buf)
            if total and (last // (1 << 20)) % 16 == 0:
                print(f"  {last/1048576:6.1f} / {total/1048576:.1f} MB  "
                      f"{last/1048576/max(time.time()-t0,1e-6):5.2f} MB/s", flush=True)
    os.replace(tmp, DST)
    print(f"done: {DST}  {size_of(DST)/1048576:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    download(ap.parse_args().force)

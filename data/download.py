"""download.py — 下载 Arena 人类偏好数据 (v2 阶段 1, 第 0 步)

决策与理由
----------
1. 数据源: lmarena-ai/arena-human-preference-140k (2025-04 ~ 2025-07, 135,634 条纯文本对战)。
   - 原计划的 lmsys/arena-human-preference-100k (2024 年模型) 和 55k 都比它旧。
     Yiqing 2026-11-07 决定只用最新的 140k, 旧版不再保留。
   - lmsys/ 组织已改名 lmarena-ai/, 旧地址会跳转或 401。
2. 只下 data/ 下的 7 个 parquet 分片, 总共约 1.6GB。
3. 下载物视为"不可信文件": 放进单独目录 (不和脚本混放), 下载后逐个用 HF 公布的
   LFS sha256 校验, 保证拿到的是官方原文件, 没被截断或篡改。不一致就删掉并报错。
4. 只用标准库 (urllib), 不引入 huggingface_hub, 依赖越少越好复现。
5. 网络: 云端环境需放行 huggingface.co 和 us.aws.cdn.hf.co (文件本体从后者下发)。

用法: python -I download.py <输出目录>
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = "lmarena-ai/arena-human-preference-140k"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/data"
FILE = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"


def list_files():
    """从 HF API 拿分片列表和每个文件的官方 sha256 (LFS oid)。"""
    with urllib.request.urlopen(API, timeout=60) as r:
        return [(x["path"], x["lfs"]["oid"]) for x in json.load(r) if x["path"].endswith(".parquet")]


def download(path: str, sha: str, out_dir: Path) -> Path:
    dest = out_dir / Path(path).name
    h = hashlib.sha256()
    with urllib.request.urlopen(FILE.format(path=path), timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
    if h.hexdigest() != sha:
        dest.unlink()
        raise ValueError(f"sha256 不一致, 已删除 {dest}")
    return dest


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_files()
    for path, sha in files:
        print(f"OK {download(path, sha, out_dir)} (sha256 校验通过)")
    (out_dir / "SOURCE.txt").write_text(
        f"repo={REPO}\n" + "".join(f"{p} sha256={s}\n" for p, s in files))


if __name__ == "__main__":
    main()

"""Assemble the Hugging Face Space bundle in dist/space (gitignored), ready to `git push` to a Space repo.

    python scripts/make_space.py
"""
import gzip
import os
import shutil
import sqlite3

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "dist", "space")

if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(os.path.join(OUT, "data", "prepared"))
os.makedirs(os.path.join(OUT, "results", "cache"))
os.makedirs(os.path.join(OUT, "demo"))

for f in ("Dockerfile", "start.sh", "README.md"):
    shutil.copy(os.path.join(ROOT, "space", f), os.path.join(OUT, f))
# start.sh must keep LF endings inside a Linux image
p = os.path.join(OUT, "start.sh")
data = open(p, "rb").read().replace(b"\r\n", b"\n")
with open(p, "wb") as f:
    f.write(data)

shutil.copytree(os.path.join(ROOT, "desirepath"), os.path.join(OUT, "desirepath"),
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy(os.path.join(ROOT, "demo", "app.py"), os.path.join(OUT, "demo", "app.py"))
shutil.copy(os.path.join(ROOT, "results", "curve.png"), os.path.join(OUT, "results", "curve.png"))
shutil.copy(os.path.join(ROOT, "results", "demo_questions.json"), os.path.join(OUT, "results", "demo_questions.json"))

# data files over 10 MB need LFS on the Hub, so ship them gzipped (start.sh gunzips)
for name in ("entities.json", "kb.json", "wiki.json", "workload.json"):
    src = os.path.join(ROOT, "data", "prepared", name)
    with open(src, "rb") as fi, gzip.open(os.path.join(OUT, "data", "prepared", name + ".gz"), "wb", 9) as fo:
        shutil.copyfileobj(fi, fo)

# consistent snapshot of the recorded Claude cache (WAL-safe)
src = sqlite3.connect(os.path.join(ROOT, "results", "cache", "llm.sqlite"))
dst = sqlite3.connect(os.path.join(OUT, "results", "cache", "llm.sqlite"))
src.backup(dst)
dst.execute("PRAGMA journal_mode=DELETE")
dst.commit()
n = dst.execute("SELECT count(*) FROM cache").fetchone()[0]
dst.close()
src.close()

total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
print(f"bundle at {OUT}: {total / 1e6:.1f} MB, {n} recorded Claude responses")

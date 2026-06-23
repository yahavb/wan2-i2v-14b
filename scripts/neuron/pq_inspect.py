"""Print columns + sample rows of the DMA-relevant parquet tables from a
neuron-explorer `view --output-format parquet` dump. Uses duckdb (no pandas).

Usage: python3 pq_inspect.py <parquet-dir>
"""
import sys
import os
import duckdb

pqdir = sys.argv[1]
tables = ["DmaPacketAggregated", "OpcodeSummary", "HloInstruction",
          "DmaPacket", "Summary", "Instruction"]

con = duckdb.connect()
for t in tables:
    path = os.path.join(pqdir, t + ".parquet")
    if not os.path.exists(path):
        print("\n===== %s : MISSING =====" % t)
        continue
    n = con.execute("SELECT COUNT(*) FROM read_parquet(?)", [path]).fetchone()[0]
    cols = [c[0] for c in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [path]).fetchall()]
    print("\n===== %s : %d rows =====" % (t, n))
    print("cols:", cols)
    rows = con.execute("SELECT * FROM read_parquet(?) LIMIT 3", [path]).fetchall()
    for r in rows:
        print(r)

"""Print columns + sample rows of the DMA-relevant parquet tables from a
neuron-explorer `view --output-format parquet` dump.

Usage: python3 pq_inspect.py <parquet-dir>
"""
import sys
import os
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

pqdir = sys.argv[1]
tables = ["DmaPacketAggregated", "OpcodeSummary", "HloInstruction",
          "DmaPacket", "Summary", "Instruction"]

for t in tables:
    path = os.path.join(pqdir, t + ".parquet")
    if not os.path.exists(path):
        print("\n===== %s : MISSING =====" % t)
        continue
    df = pd.read_parquet(path)
    print("\n===== %s : %d rows =====" % (t, len(df)))
    print("cols:", list(df.columns))
    print(df.head(3).to_string())

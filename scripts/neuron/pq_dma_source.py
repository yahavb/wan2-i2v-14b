"""Locate the source of the software-dynamic (gather/indirect) DMA in a
neuron-explorer parquet dump. Uses only verified columns. duckdb, no pandas.

Usage: python3 pq_dma_source.py <parquet-dir>
"""
import sys
import os
import duckdb

pq = sys.argv[1]
DPA = os.path.join(pq, "DmaPacketAggregated.parquet")
INS = os.path.join(pq, "Instruction.parquet")
con = duckdb.connect()


def show(title, sql):
    print("\n===== %s =====" % title)
    try:
        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description]
        print(" | ".join(cols))
        for r in rows:
            print(" | ".join("" if v is None else str(v) for v in r))
    except Exception as e:
        print("query error:", e)


# 1. DMA aggregated by queue/type/op/direction — where do the bytes go?
show("DMA by queue/type/direction (top by bytes)", f"""
  SELECT queue_type, queue_name, op, source, dest, is_transpose_mode,
         COUNT(*) n, SUM(transfer_bytes) bytes, SUM(duration_ns) dur_ns
  FROM read_parquet('{DPA}')
  GROUP BY 1,2,3,4,5,6 ORDER BY dur_ns DESC LIMIT 20
""")

# 2. Does ANY instruction carry source attribution? (debug info present?)
show("Instruction source-attribution coverage", f"""
  SELECT
    COUNT(*) total,
    COUNT(nki_source_location)               nki_nn,
    COUNT(bir_debug_info_source_location)    bir_nn,
    COUNT(hlo_name)                          hlo_nn,
    COUNT(CASE WHEN dma_trigger_start_ts IS NOT NULL THEN 1 END) dma_trig
  FROM read_parquet('{INS}')
""")

# 3. The heaviest instructions by total duration — engine/opcode + any source.
show("Top instructions by duration (engine/opcode + source)", f"""
  SELECT engine, opcode,
         COUNT(*) n, SUM(duration_ns) dur_ns, SUM(dma_wait_time_ns) dma_wait_ns,
         ANY_VALUE(nki_source_location) nki_src,
         ANY_VALUE(bir_debug_info_source_location) bir_src,
         ANY_VALUE(hlo_name) hlo
  FROM read_parquet('{INS}')
  GROUP BY 1,2 ORDER BY dur_ns DESC LIMIT 20
""")

# 4. Instructions that TRIGGER dma, grouped by whatever source we have.
show("DMA-triggering instructions by source", f"""
  SELECT engine, opcode,
         COALESCE(NULLIF(nki_source_location,''),
                  NULLIF(bir_debug_info_source_location,''),
                  NULLIF(hlo_name,''),
                  NULLIF(bir_instruction_name,''), '(no source)') src,
         COUNT(*) n, SUM(dma_wait_time_ns) dma_wait_ns, SUM(duration_ns) dur_ns
  FROM read_parquet('{INS}')
  WHERE dma_trigger_start_ts IS NOT NULL
  GROUP BY 1,2,3 ORDER BY dma_wait_ns DESC LIMIT 25
""")

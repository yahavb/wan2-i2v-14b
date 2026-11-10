#!/usr/bin/env bash
# The accuracy gate for every candidate: CPU fp32 rollout reference R, base B = the approved
# 2-device config (trn3 TP8, validated stage), candidate C. Exit 0 = PASS.
#   bash scripts/validate/gate_wan.sh <R_dir> <B_gate_dir> <C_gate_dir> [verdict.json]
# Gate dirs come from scripts/validate/dump_to_gate.py (WAN_DUMP_DIR dump); R from ref_rollout_fp32.py.
#
# Thresholds calibrated on user-labelled videos (2026-11-10, 368x480 21f 10 steps, vs R; all 8
# shown were approved: "all looks amazing"; no rejected wan2 video yet, so the floors sit below
# the worst approved one with margin; o2's rejected (blurred) video had sharpness 0.17):
#   approved   first-9-frame PSNR 22.35-25.48 dB   whole PSNR 22.01-25.54   worst frame 21.11-23.84
#              sharpness/R 0.955-1.011              step-0 rel-RMS 0.0352-0.0400
#   floors     early PSNR >= 21.5, step-0 rel <= 0.045, sharpness >= 0.90, whole PSNR >= 21.5,
#              G1/G2 within 2.5 dB of B (B 24.19 / 22.80; exact-equivalent configs spread 1.5-3.5 dB
#              because 10 bf16 steps amplify accumulation order), G3 step-0 rel <= 1.3 x B.
set -u
R=$1 B=$2 CAND=$3 J=${4:-}
python3 "$(dirname "$0")/../validate_gate.py" --mode pixels --ref "$R" --base "$B" --cand "$CAND" \
  --floor-early-db 21.5 --early-frames 9 --max-early-rel 0.045 --min-sharpness 0.90 \
  --floor-db 21.5 --delta-db 2.5 --delta-worst-db 2.5 --rel-ratio 1.3 ${J:+--json-out "$J"}

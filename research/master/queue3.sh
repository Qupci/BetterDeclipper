#!/bin/bash
cd /f/BetterDeclipper
PY=.venv/Scripts/python.exe
until grep -q "QUEUE2 DONE" research/results/fusion_multi.log 2>/dev/null; do sleep 30; done
$PY research/master/run_bench.py dip dip:2.0 dip:3.0 dip:5.0 dip:3.0:b0.5 --degs al1_6,osal1 > research/results/mbench_dip.log 2>&1
echo QUEUE3 DONE >> research/results/mbench_dip.log

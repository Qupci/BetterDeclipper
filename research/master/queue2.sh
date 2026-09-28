#!/bin/bash
cd /f/BetterDeclipper
PY=.venv/Scripts/python.exe
until grep -q "QUEUE1 DONE" research/results/mbench_raise.log 2>/dev/null; do sleep 30; done
$PY research/master/run_bench.py prox prox:0.25 prox:0.5 prox:1.0 proxu:0.5 proxu:1.0 proxk:0.5:0.8 proxk:1.0:0.8 --degs al1_6,osal1 > research/results/mbench_prox.log 2>&1
$PY research/fusion/make_components_multi.py > research/results/fusion_multi.log 2>&1
echo QUEUE2 DONE >> research/results/fusion_multi.log

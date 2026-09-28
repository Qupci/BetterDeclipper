#!/bin/bash
cd /f/BetterDeclipper
PY=.venv/Scripts/python.exe
# wait for the smear sweep to finish
while [ "$(wc -l < research/results/mbench_smear.log)" -lt 35 ]; do sleep 20; done
$PY research/master/run_bench.py base auto hard soft soft:0.5 soft:0.6 soft:0.7 soft:0.9 --srcs thrash,violet,bigshot,knife >> research/results/mbench_base.log 2>&1
$PY research/master/run_bench.py raise raise:0.005 raise:0.01 raise:0.02 smear6 --degs hard6,hard9,tanh6,tanh9k4,cubic6,softhard6,softhard9,oshard6,resamp6,mp3_128,aac_256,vorbis_192,al1_6,osal1 > research/results/mbench_raise.log 2>&1
echo QUEUE1 DONE >> research/results/mbench_raise.log

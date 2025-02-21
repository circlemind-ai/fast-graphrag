:: 2wikimultihopqa benchmark
:: Creating databases
python graph_benchmark.py -n 51 -c
python graph_benchmark.py -n 101 -c

:: Evaluation (create reports)
python graph_benchmark.py -n 51 -b
python graph_benchmark.py -n 101 -b
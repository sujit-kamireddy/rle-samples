You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- uber-raw-data-apr14.csv
- uber-raw-data-may14.csv
- uber-raw-data-jun14.csv
- uber-raw-data-jul14.csv
- uber-raw-data-aug14.csv
- uber-raw-data-sep14.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
Is the original raw time series (number of Uber pickups per hour) stationary according to the ADF test at a 5% significance level?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
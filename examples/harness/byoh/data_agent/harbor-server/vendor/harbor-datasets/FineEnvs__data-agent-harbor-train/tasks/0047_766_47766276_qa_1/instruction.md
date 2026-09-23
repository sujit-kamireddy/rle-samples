You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- data.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
What percentage of the data corresponds to each season (winter, spring, summer, autumn)?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as comma-separated <season>:<percentage> pairs, with the season label first and the percentage as a plain number (e.g. 24.30), in the order <label4>, <label3>, <label1>, <label2>.

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- degrees-that-pay-back.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
What is the most common range for the percentage change from starting to mid-career salary based on the histogram visualization?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as: <range> (e.g., 60% <label1>%), with the lower and upper bounds as percentages separated by " to ".

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
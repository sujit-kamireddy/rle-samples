You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- german_credit_data.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
How many missing values existed in the "Saving accounts" and "Checking account" columns before imputation, and what strategy was used to fill them?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as: <label2>: <count>, <label1>: <count>, <strategy> (comma-separated, in that order, counts as plain numbers).

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
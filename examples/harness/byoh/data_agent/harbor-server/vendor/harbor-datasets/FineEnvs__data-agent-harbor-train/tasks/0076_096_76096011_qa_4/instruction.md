You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- UCI_Credit_Card.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
What is the proportion of clients who did not default compared to those who defaulted in the dataset, as shown in the exploratory data analysis?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as: <percentage> <label1>, <percentage> <label2> (comma-separated, percentages with % sign, order as given).

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
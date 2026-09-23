You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- Iris.csv
- database.sqlite

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
What are the Petal Length and Petal Width values of the cluster centroid with the highest Petal Length in the 5-cluster K-means model?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as: <petal_length>, <petal_width> (comma-separated, numeric values with two decimals).

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
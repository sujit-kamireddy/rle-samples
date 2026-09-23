You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- mtcars.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
What is the average miles per gallon (mpg) for manual transmission cars compared to automatic transmission cars in the dataset?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as: <transmission type>: <average mpg> pairs, comma-separated, with the transmission type label first and the mpg value as a plain number with two decimals.

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
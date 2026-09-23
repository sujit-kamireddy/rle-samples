You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- sign_mnist_train.csv
- sign_mnist_test.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
Which alphabetic character corresponding to a label is missing in the classes mapping?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
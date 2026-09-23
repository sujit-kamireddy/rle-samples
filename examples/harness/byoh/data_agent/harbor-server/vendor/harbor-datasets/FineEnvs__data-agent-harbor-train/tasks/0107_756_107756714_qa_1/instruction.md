You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- training.1600000.processed.noemoticon.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
What is the proportion of positive and negative sentiment classes in the dataset before any resampling?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as a comma-separated list of <class>: <percentage> pairs, with the class label first and the percentage as a plain number (e.g., 50), not a fraction.

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
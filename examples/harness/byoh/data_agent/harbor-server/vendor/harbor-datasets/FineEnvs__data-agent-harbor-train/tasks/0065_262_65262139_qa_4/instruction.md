You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- training.1600000.processed.noemoticon.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
How many tweets were allocated to the training set and testing set after applying an 80-20 train/test split on the 200,000-sample subset of the dataset?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as a comma-separated pair: <training_count>, <testing_count> (plain numbers, no labels).

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
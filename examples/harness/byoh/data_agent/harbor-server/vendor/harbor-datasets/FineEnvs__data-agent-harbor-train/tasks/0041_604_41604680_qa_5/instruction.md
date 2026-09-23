You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- Train.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
What numerical values are assigned to the "Low Fat" and "Regular" categories when using LabelEncoder on the cleaned Item_Fat_Content column?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as: <category> = <value>, <category> = <value> (comma-separated, category first, plain numbers).

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
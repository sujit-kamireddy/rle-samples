You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- haberman.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
What is the median number of positive lymph nodes for patients who survived (1) compared to those who did not survive (0)?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as: <group>, <value> pairs, comma-separated, with group labels "Survived" and "<label1>" first, followed by the median as a plain integer.

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
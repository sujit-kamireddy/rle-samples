You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- diabetes.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
What is the percentage distribution of patients in the defined age groups (Young: 29-40, Middle: 40-55, Elderly: ≥55 years)?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as a comma-separated list of <age group>: <percentage> pairs, in the order <label3>, <label2>, <label1>, with percentages as numbers (e.g., 48.38).

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
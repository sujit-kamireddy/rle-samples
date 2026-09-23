You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- WA_Fn-UseC_-Telco-Customer-Churn.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
What is the churn rate percentage for customers with 'Fiber optic' internet service compared to 'DSL' customers?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

: Answer as: <service>, <percentage> pairs, comma-separated, with the service name first and the percentage as a plain number with two decimals followed by a percent sign (e.g., <label1>: 41.89%, <label2>: 18.96%).

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
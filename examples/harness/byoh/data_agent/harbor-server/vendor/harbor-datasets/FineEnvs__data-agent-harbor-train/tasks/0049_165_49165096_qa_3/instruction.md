You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- HN_posts_year_to_Sep_26_2016.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
What are the top 5 hours of the day for 'ask' posts in terms of average comments, listed from highest to lowest average?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as a comma-separated list of hours, from highest to lowest average, using two-digit 24-hour format (e.g., 15, 13, 12, 02, 10).

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
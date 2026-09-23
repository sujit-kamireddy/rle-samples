You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- Social_Network_Ads.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
Which K value in the KNN model using raw data achieved the highest accuracy, and what was that accuracy score?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as: <K value>, <accuracy> (comma-separated, K value first, accuracy as a fraction e.g. 0.89).

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
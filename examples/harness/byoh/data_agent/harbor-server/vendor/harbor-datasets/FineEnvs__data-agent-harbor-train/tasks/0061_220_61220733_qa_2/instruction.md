You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- nba_2017_team_valuations.csv
- nba_2017_elo.csv
- nba_2017_att_val_elo.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
Which team had the highest total attendance in the 2017 season and what was its valuation?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as: <team>, <valuation> (comma-separated, team first, plain number).

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
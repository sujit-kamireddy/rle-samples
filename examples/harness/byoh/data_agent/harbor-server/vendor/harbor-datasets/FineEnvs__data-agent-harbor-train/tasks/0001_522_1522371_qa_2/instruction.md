You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- mls-salaries-2007.csv
- mls-salaries-2008.csv
- mls-salaries-2009.csv
- mls-salaries-2010.csv
- mls-salaries-2011.csv
- mls-salaries-2012.csv
- mls-salaries-2013.csv
- mls-salaries-2014.csv
- mls-salaries-2015.csv
- mls-salaries-2016.csv
- mls-salaries-2017.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
Who was the highest-paid player in the 2017 season, and what was their guaranteed compensation amount?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as: <player name>, <amount> (comma-separated, name first, plain number).

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
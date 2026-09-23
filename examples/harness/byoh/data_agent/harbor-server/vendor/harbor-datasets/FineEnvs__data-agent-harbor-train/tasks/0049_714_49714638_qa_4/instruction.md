You are a data-analysis agent working in a sandbox. Use your code-execution tool to inspect the files and compute the answer.

Files (in /home/user/input, no subfolders):
- pokemon.csv
- combats.csv
- tests.csv

Installed: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels, tabulate, sqlite3, plotly (pip install more if needed).

Question:
Which Pokémon have the highest Defense points, and what is their specific Defense value?

Work it out step by step — inspect the data first (head, shape, dtypes), then compute.

Answer as a comma-separated list of the exact Pokémon names, followed by a hyphen and the Defense value (e.g., Name1, Name2, Name3 - 230).

Answer with a single clean value: a bare number (no commas or units, e.g. 95293), a short label, yes/no, or a comma-separated list. Keep decimal precision. If there's no applicable answer, write: Not Applicable

Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`), then stop.
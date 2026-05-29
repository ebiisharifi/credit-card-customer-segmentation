"""
Loads the customer table (with cluster labels) into an in-memory SQLite
database and runs every query in customer_segments.sql, printing the results.

Run it after analysis.py (which creates customers_with_clusters.csv):
    python sql/run_sql.py
"""
import os
import re
import sqlite3
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# the clustered table does not keep CUST_ID (it is dropped before modelling),
# so I add it back from the raw file - the row order is the same.
clustered = pd.read_csv(os.path.join(ROOT, "data", "customers_with_clusters.csv"))
raw = pd.read_csv(os.path.join(ROOT, "data", "credit_card_customers.csv"))
clustered.insert(0, "CUST_ID", raw["CUST_ID"].values)

con = sqlite3.connect(":memory:")
clustered.to_sql("customers", con, index=False)

# read the .sql file and split it into separate statements
with open(os.path.join(HERE, "customer_segments.sql"), encoding="utf-8") as f:
    script = f.read()

statements = [s.strip() for s in script.split(";") if s.strip() and "SELECT" in s.upper()]

for i, stmt in enumerate(statements, 1):
    # grab the first real comment line as a title
    title = next((ln.strip("- ").strip() for ln in stmt.splitlines()
                  if ln.strip().startswith("--") and re.search("[a-zA-Z]", ln.strip("- ="))),
                 f"query {i}")
    print("\n" + "=" * 70)
    print(f"Q{i}: {title}")
    print("=" * 70)
    print(pd.read_sql_query(stmt, con).to_string(index=False))

con.close()

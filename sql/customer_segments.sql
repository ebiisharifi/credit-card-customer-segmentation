-- ===========================================================================
-- Customer segmentation - SQL exploration
-- ---------------------------------------------------------------------------
-- The clustering itself is done in Python (it needs K-Means), but a lot of the
-- profiling and the "so what" questions are easy to answer in plain SQL. This
-- is the kind of query I would run if the customer data and the cluster labels
-- were sitting in a database table.
--
-- Table:  customers   (one row per cardholder, plus a `cluster` column)
-- I load it from data/customers_with_clusters.csv  (see run_sql.py).
-- Written for SQLite, but it is standard SQL and runs almost as-is elsewhere.
-- ===========================================================================


-- 1. how many customers in total, and how big is each segment
SELECT
    cluster,
    COUNT(*)                                  AS n_customers,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_base
FROM customers
GROUP BY cluster
ORDER BY n_customers DESC;


-- 2. the behaviour profile of each segment (this is the main table)
SELECT
    cluster,
    COUNT(*)                          AS n,
    ROUND(AVG(BALANCE), 0)            AS avg_balance,
    ROUND(AVG(PURCHASES), 0)          AS avg_purchases,
    ROUND(AVG(CASH_ADVANCE), 0)       AS avg_cash_advance,
    ROUND(AVG(PAYMENTS), 0)           AS avg_payments,
    ROUND(AVG(PRC_FULL_PAYMENT), 2)   AS avg_full_payment_rate,
    ROUND(AVG(CREDIT_LIMIT), 0)       AS avg_credit_limit
FROM customers
GROUP BY cluster
ORDER BY avg_cash_advance DESC;


-- 3. a simple risk flag without any model - just business rules.
--    high balance AND heavy cash advance AND almost never pays in full.
--    useful to sanity-check that the "risky" clusters really are risky.
SELECT
    cluster,
    COUNT(*) AS n_flagged
FROM customers
WHERE BALANCE > 2000
  AND CASH_ADVANCE > 1000
  AND PRC_FULL_PAYMENT < 0.1
GROUP BY cluster
ORDER BY n_flagged DESC;


-- 4. who are the most valuable customers (by total purchases) and where do
--    they sit. classic "top N" question a business would ask.
SELECT
    CUST_ID,
    cluster,
    ROUND(PURCHASES, 0) AS purchases,
    ROUND(PAYMENTS, 0)  AS payments
FROM customers
ORDER BY PURCHASES DESC
LIMIT 10;


-- 5. bucket customers by how active they are on purchases, regardless of
--    cluster. shows a CASE / segmentation done directly in SQL.
SELECT
    CASE
        WHEN PURCHASES = 0               THEN '0 - never purchases'
        WHEN PURCHASES < 500             THEN '1 - light'
        WHEN PURCHASES < 2000            THEN '2 - medium'
        ELSE                                  '3 - heavy'
    END                                  AS purchase_band,
    COUNT(*)                             AS n_customers,
    ROUND(AVG(PRC_FULL_PAYMENT), 2)      AS avg_full_payment_rate
FROM customers
GROUP BY purchase_band
ORDER BY purchase_band;

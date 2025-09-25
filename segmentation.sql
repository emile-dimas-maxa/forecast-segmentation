-- =====================================================
-- EOM FORECASTING FEATURE ENGINEERING SQL
-- DUAL PATTERN SEGMENTATION (EOM vs GENERAL TIMESERIES)
-- Version 4.0 - Separate EOM and General Pattern Analysis
-- =====================================================
-- This script generates features using rolling windows to prevent data leakage
-- Separate behavioral patterns for EOM-specific and general timeseries
-- Independent thresholds for overall vs EOM importance tiers
-- Snowflake compatible - No nested window functions in same CTE

-- Configuration parameters
WITH config AS (
    SELECT 
        -- Basic configuration
        3 AS min_months_history,      -- Minimum months of history required
        12 AS rolling_window_months,   -- Rolling window for feature calculation
        6 AS min_transactions,         -- Minimum non-zero transactions to include series
        '2022-01-01'::DATE AS start_date,  -- Analysis start date
        CURRENT_DATE AS end_date,     -- Analysis end date
        
        -- Window sizes for rolling calculations
        3 AS ma_window_short,         -- Short-term moving average window (months)
        6 AS pre_eom_signal_window,   -- Pre-EOM signal rolling window (months)
        5 AS pre_eom_days,            -- Days before EOM to consider for pre-EOM signals
        
        -- Calendar day thresholds
        10 AS early_month_days,       -- First N days of month for early month signal
        20 AS mid_month_end_day,      -- End day for mid-month period (10-20)
        
        -- Overall importance thresholds (based on total volume)
        100000000000 AS critical_volume_threshold,      -- Critical overall volume (12 months)
        5000000000 AS high_volume_threshold,           -- High overall volume (12 months)
        1000000000 AS medium_volume_threshold,         -- Medium overall volume (12 months)
        
        -- Monthly average thresholds (derived from annual)
        1000000000 AS critical_monthly_avg_threshold,   -- Critical monthly average
        500000000 AS high_monthly_avg_threshold,       -- High monthly average
        100000000 AS medium_monthly_avg_threshold,      -- Medium monthly average
        
        -- Max single transaction thresholds
        50000000 AS critical_max_transaction_threshold,  -- Critical single transaction
        10000000 AS high_max_transaction_threshold,       -- High single transaction
        5000000 AS medium_max_transaction_threshold,     -- Medium single transaction
        
        -- EOM importance thresholds (based on EOM-specific volume)
        50000000000 AS critical_eom_volume_threshold,   -- Critical EOM volume (12 months)
        50000000000 AS high_eom_volume_threshold,       -- High EOM volume (12 months)
        50000000000 AS medium_eom_volume_threshold,      -- Medium EOM volume (12 months)
        
        -- EOM monthly average thresholds
        50000000000 AS critical_eom_monthly_threshold,   -- Critical EOM monthly average
        50000000000 AS high_eom_monthly_threshold,       -- High EOM monthly average
        50000000000 AS medium_eom_monthly_threshold,      -- Medium EOM monthly average
        
        -- Max single EOM transaction thresholds
        100000000 AS critical_max_eom_threshold,      -- Critical single EOM transaction
        50000000 AS high_max_eom_threshold,          -- High single EOM transaction
        10000000 AS medium_max_eom_threshold,         -- Medium single EOM transaction
        
        -- SEPARATE Portfolio percentile thresholds for OVERALL importance
        0.2 AS overall_critical_percentile,    -- Top 10% for overall
        0.4 AS overall_high_percentile,        -- Top 30% for overall
        0.8 AS overall_medium_percentile,      -- Top 70% for overall
        
        -- SEPARATE Portfolio percentile thresholds for EOM importance
        0.3 AS eom_critical_percentile,    -- Top 15% for EOM (can be different)
        0.6 AS eom_high_percentile,        -- Top 35% for EOM
        0.95 AS eom_medium_percentile,      -- Top 75% for EOM
        
        -- EOM-specific pattern thresholds
        0.70 AS eom_high_concentration_threshold,      -- High EOM concentration
        0.40 AS eom_medium_concentration_threshold,    -- Medium EOM concentration
        0.70 AS eom_high_predictability_threshold,     -- High EOM predictability
        0.40 AS eom_medium_predictability_threshold,   -- Medium EOM predictability
        0.50 AS eom_high_frequency_threshold,          -- High EOM frequency
        0.30 AS eom_intermittent_threshold,            -- Intermittent EOM threshold
        
        -- General timeseries pattern thresholds
        0.50 AS ts_high_volatility_threshold,          -- High volatility (CV)
        0.25 AS ts_medium_volatility_threshold,        -- Medium volatility (CV)
        0.70 AS ts_high_regularity_threshold,          -- High transaction regularity
        0.40 AS ts_medium_regularity_threshold,        -- Medium transaction regularity
        0.30 AS ts_intermittent_threshold,             -- Intermittent activity threshold
        0.25 AS ts_seasonal_concentration_threshold,   -- Quarterly concentration for seasonality
        
        -- Other thresholds
        3 AS inactive_months,                      -- Months of inactivity for INACTIVE
        3 AS emerging_months,                      -- Max months for EMERGING
        100000 AS eom_risk_volume_threshold,       -- Volume threshold for EOM risk flag
        6 AS eom_risk_min_months,                  -- Min months history for risk assessment
        
        -- Trend analysis thresholds
        6 AS trend_analysis_months,        -- Months required for trend analysis
        5 AS trend_window_months,          -- Window for trend calculation (preceding months)
        0.1 AS growth_threshold,           -- Growth rate threshold for trend classification
        12 AS yoy_comparison_months        -- Months for year-over-year comparison
)

-- =====================================================
-- STEP 1: DATA PREPARATION
-- =====================================================

, t__core_banking_time_series AS (
    select *, from {{ ref('int__t__cad_core_banking_regular_time_series_recorded') }}
)

, base_data AS (
    SELECT 
        dim_value,
        date,
        amount,
        is_last_work_day_of_month,
        -- Add month and quarter identifiers
        DATE_TRUNC('month', date) AS month,
        DATE_TRUNC('quarter', date) AS quarter,
        EXTRACT(YEAR FROM date) AS year,
        EXTRACT(MONTH FROM date) AS month_num,
        EXTRACT(QUARTER FROM date) AS quarter_num,
        -- Add day position in month
        EXTRACT(DAY FROM date) AS day_of_month,
        -- Business days from EOM (negative = before EOM)
        CASE 
            WHEN is_last_work_day_of_month = TRUE THEN 0
            ELSE -ROW_NUMBER() OVER (
                PARTITION BY dim_value, DATE_TRUNC('month', date) 
                ORDER BY date DESC
            )
        END AS days_from_eom
    FROM 
        t__core_banking_time_series
    WHERE 
        date >= (SELECT start_date FROM config)
        AND date <= (SELECT end_date FROM config)
)

-- =====================================================
-- STEP 2: MONTHLY AGGREGATIONS
-- =====================================================

, monthly_aggregates AS (
    SELECT 
        dim_value,
        month,
        year,
        month_num,
        
        -- Total monthly amounts
        COALESCE(SUM(amount), 0) AS monthly_total,
        COUNT(CASE WHEN amount <> 0 THEN 1 ELSE NULL END) AS monthly_transactions,
        COALESCE(AVG(CASE WHEN amount <> 0 THEN amount END), 0) AS monthly_avg_amount,
        COALESCE(STDDEV_POP(CASE WHEN amount <> 0 THEN amount END), 0) AS monthly_std_amount,
        
        -- EOM specific amounts
        COALESCE(SUM(CASE WHEN is_last_work_day_of_month THEN amount ELSE 0 END), 0) AS eom_amount,
        COUNT(CASE WHEN is_last_work_day_of_month AND amount <> 0 THEN 1 ELSE NULL END) AS eom_transaction_count,
        
        -- Non-EOM amounts
        COALESCE(SUM(CASE WHEN NOT is_last_work_day_of_month THEN amount ELSE 0 END), 0) AS non_eom_total,
        COUNT(CASE WHEN NOT is_last_work_day_of_month AND amount <> 0 THEN 1 ELSE NULL END) AS non_eom_transactions,
        COALESCE(AVG(CASE WHEN NOT is_last_work_day_of_month AND amount <> 0 THEN amount END), 0) AS non_eom_avg,
        
        -- Pre-EOM signals
        COALESCE(SUM(CASE WHEN days_from_eom BETWEEN -(SELECT pre_eom_days FROM config) AND -1 THEN amount ELSE 0 END), 0) AS pre_eom_5d_total,
        COUNT(CASE WHEN days_from_eom BETWEEN -(SELECT pre_eom_days FROM config) AND -1 AND amount <> 0 THEN 1 ELSE NULL END) AS pre_eom_5d_count,
        
        -- Early month signal
        COALESCE(SUM(CASE WHEN day_of_month <= (SELECT early_month_days FROM config) THEN amount ELSE 0 END), 0) AS early_month_total,
        
        -- Mid month signal
        COALESCE(SUM(CASE WHEN day_of_month BETWEEN (SELECT early_month_days FROM config) AND (SELECT mid_month_end_day FROM config) THEN amount ELSE 0 END), 0) AS mid_month_total,
        
        -- Maximum single transaction
        COALESCE(MAX(amount), 0) AS max_monthly_transaction,
        
        -- Transaction distribution within month (how spread out are transactions)
        COALESCE(STDDEV_POP(day_of_month), 0) AS day_dispersion,
        
        -- Quarter-end indicator
        MAX(CASE WHEN month_num IN (3, 6, 9, 12) THEN 1 ELSE 0 END) AS is_quarter_end,
        
        -- Year-end indicator
        MAX(CASE WHEN month_num = 12 THEN 1 ELSE 0 END) AS is_year_end,
        
        -- EOM activity flag
        MAX(CASE WHEN is_last_work_day_of_month AND amount > 0 THEN 1 ELSE 0 END) AS has_nonzero_eom,
        
        -- Quarterly totals for seasonality detection
        SUM(CASE WHEN month_num IN (3, 6, 9, 12) THEN amount ELSE 0 END) AS quarter_end_amount,
        SUM(CASE WHEN month_num = 12 THEN amount ELSE 0 END) AS year_end_amount
        
    FROM 
        base_data
    GROUP BY 
        dim_value, month, year, month_num
)

-- =====================================================
-- STEP 3: ROLLING WINDOW FEATURES
-- =====================================================

, rolling_features AS (
    SELECT 
        ma.dim_value,
        ma.month,
        ma.year,
        ma.month_num,
        ma.monthly_total,
        ma.monthly_transactions,
        ma.eom_amount,
        ma.non_eom_total,
        ma.has_nonzero_eom,
        ma.is_quarter_end,
        ma.is_year_end,
        ma.quarter_end_amount,
        ma.year_end_amount,
        ma.day_dispersion,
        
        -- Overall volume metrics (12-month rolling)
        SUM(ma.monthly_total) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_total_volume_12m,
        
        AVG(ma.monthly_total) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_avg_monthly_volume,
        
        MAX(ma.max_monthly_transaction) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_max_transaction,
        
        -- EOM-specific metrics
        SUM(ma.eom_amount) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_eom_volume_12m,
        
        AVG(CASE WHEN ma.eom_amount > 0 THEN ma.eom_amount END) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_avg_nonzero_eom,
        
        MAX(ma.eom_amount) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_max_eom,
        
        STDDEV_POP(ma.eom_amount) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_std_eom,
        
        -- Non-EOM metrics (for general timeseries patterns)
        SUM(ma.non_eom_total) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_non_eom_volume_12m,
        
        AVG(ma.non_eom_total) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_avg_non_eom,
        
        -- Frequency counts
        SUM(CASE WHEN ma.eom_amount > 0 THEN 1 ELSE 0 END) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_nonzero_eom_months,
        
        SUM(CASE WHEN ma.eom_amount = 0 THEN 1 ELSE 0 END) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_zero_eom_months,
        
        SUM(CASE WHEN ma.monthly_total > 0 THEN 1 ELSE 0 END) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS active_months_12m,
        
        -- Volatility metrics
        STDDEV_POP(ma.monthly_total) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_std_monthly,
        
        -- Seasonality metrics
        SUM(ma.quarter_end_amount) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_quarter_end_volume,
        
        SUM(ma.year_end_amount) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_year_end_volume,
        
        -- Transaction regularity
        AVG(ma.monthly_transactions) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_avg_transactions,
        
        STDDEV_POP(ma.monthly_transactions) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_std_transactions,
        
        -- Day dispersion (for pattern detection)
        AVG(ma.day_dispersion) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS rolling_avg_day_dispersion,
        
        -- Total counts (expanding window for history)
        SUM(CASE WHEN ma.eom_amount > 0 THEN 1 ELSE 0 END) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS total_nonzero_eom_count,
        
        -- Lagged values for forecasting
        LAG(ma.eom_amount, 12) OVER (PARTITION BY ma.dim_value ORDER BY ma.month) AS eom_amount_12m_ago,
        LAG(ma.eom_amount, 3) OVER (PARTITION BY ma.dim_value ORDER BY ma.month) AS eom_amount_3m_ago,
        LAG(ma.eom_amount, 1) OVER (PARTITION BY ma.dim_value ORDER BY ma.month) AS eom_amount_1m_ago,
        
        -- Moving averages
        AVG(ma.eom_amount) OVER (
            PARTITION BY ma.dim_value 
            ORDER BY ma.month 
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS eom_ma3,
        
        -- Months of history
        ROW_NUMBER() OVER (PARTITION BY ma.dim_value ORDER BY ma.month) AS months_of_history,
        
        -- Months since last EOM activity (for recency calculation)
        CASE 
            WHEN ma.has_nonzero_eom = 1 THEN 0
            ELSE ROW_NUMBER() OVER (
                PARTITION BY ma.dim_value, ma.has_nonzero_eom 
                ORDER BY ma.month DESC
            ) - 1
        END AS months_since_last_eom
        
    FROM 
        monthly_aggregates ma
)

-- =====================================================
-- STEP 4: PORTFOLIO METRICS PER MONTH
-- =====================================================

, portfolio_metrics AS (
    SELECT 
        rf.*,
        
        -- Portfolio totals for this month
        SUM(rf.rolling_total_volume_12m) OVER (PARTITION BY rf.month) AS total_portfolio_volume,
        SUM(rf.rolling_eom_volume_12m) OVER (PARTITION BY rf.month) AS total_portfolio_eom_volume,
        
        -- Cumulative portfolio percentage for OVERALL (ordered by total volume DESC)
        SUM(rf.rolling_total_volume_12m) OVER (
            PARTITION BY rf.month
            ORDER BY rf.rolling_total_volume_12m DESC
            ROWS UNBOUNDED PRECEDING
        ) / NULLIF(SUM(rf.rolling_total_volume_12m) OVER (PARTITION BY rf.month), 0) AS cumulative_overall_portfolio_pct,
        
        -- Cumulative portfolio percentage for EOM (ordered by EOM volume DESC)
        SUM(rf.rolling_eom_volume_12m) OVER (
            PARTITION BY rf.month
            ORDER BY rf.rolling_eom_volume_12m DESC
            ROWS UNBOUNDED PRECEDING
        ) / NULLIF(SUM(rf.rolling_eom_volume_12m) OVER (PARTITION BY rf.month), 0) AS cumulative_eom_portfolio_pct
        
    FROM 
        rolling_features rf
    WHERE 
        rf.months_of_history >= 3
)

-- =====================================================
-- STEP 5: CALCULATE PATTERN METRICS
-- =====================================================

, pattern_metrics AS (
    SELECT 
        pm.*,
        
        -- EOM-specific pattern metrics
        CASE 
            WHEN pm.rolling_total_volume_12m > 0 
            THEN pm.rolling_eom_volume_12m / pm.rolling_total_volume_12m
            ELSE 0 
        END AS eom_concentration,
        
        CASE 
            WHEN pm.rolling_avg_nonzero_eom > 0 
            THEN GREATEST(0, 1 - (pm.rolling_std_eom / pm.rolling_avg_nonzero_eom)) -- Q:should it be non zero std?
            ELSE 0 
        END AS eom_predictability,
        
        CASE 
            WHEN pm.months_of_history > 1 
            THEN COALESCE(pm.rolling_nonzero_eom_months::FLOAT / LEAST(pm.months_of_history - 1, 12), 0)
            ELSE 0 
        END AS eom_frequency,
        
        CASE 
            WHEN pm.months_of_history > 1 
            THEN COALESCE(pm.rolling_zero_eom_months::FLOAT / LEAST(pm.months_of_history - 1, 12), 0)
            ELSE 0 
        END AS eom_zero_ratio,
        
        -- EOM spike detection: Compare EOM to non-EOM averages
        CASE 
            WHEN pm.rolling_avg_non_eom > 0 AND pm.rolling_avg_nonzero_eom > 0
            THEN pm.rolling_avg_nonzero_eom / pm.rolling_avg_non_eom
            WHEN pm.rolling_avg_nonzero_eom > 0 AND pm.rolling_avg_non_eom = 0
            THEN 999  -- Infinite spike (only EOM activity)
            ELSE 1
        END AS eom_spike_ratio,
        
        -- EOM consistency check: Are EOM amounts similar each time?
        CASE 
            WHEN pm.rolling_avg_nonzero_eom > 0 
            THEN pm.rolling_std_eom / NULLIF(pm.rolling_avg_nonzero_eom, 0)
            ELSE 0
        END AS eom_cv,
        
        -- General timeseries pattern metrics
        CASE 
            WHEN pm.rolling_avg_monthly_volume > 0 
            THEN pm.rolling_std_monthly / pm.rolling_avg_monthly_volume
            ELSE 0 
        END AS monthly_cv,
        
        CASE 
            WHEN pm.rolling_avg_transactions > 0 
            THEN GREATEST(0, 1 - (pm.rolling_std_transactions / pm.rolling_avg_transactions))
            ELSE 0 
        END AS transaction_regularity,
        
        CASE 
            WHEN pm.months_of_history > 0 
            THEN pm.active_months_12m::FLOAT / LEAST(pm.months_of_history, 12)
            ELSE 0 
        END AS activity_rate,
        
        -- Seasonality indicators
        CASE 
            WHEN pm.rolling_total_volume_12m > 0 
            THEN pm.rolling_quarter_end_volume / pm.rolling_total_volume_12m
            ELSE 0 
        END AS quarter_end_concentration,
        
        CASE 
            WHEN pm.rolling_total_volume_12m > 0 
            THEN pm.rolling_year_end_volume / pm.rolling_total_volume_12m
            ELSE 0 
        END AS year_end_concentration,
        
        -- Distribution pattern (concentrated vs distributed)
        CASE 
            WHEN pm.rolling_avg_day_dispersion > 0
            THEN pm.rolling_avg_day_dispersion
            ELSE 0
        END AS transaction_dispersion,
        
        -- Has EOM history flag
        CASE 
            WHEN pm.total_nonzero_eom_count > 0 
            THEN 1 
            ELSE 0 
        END AS has_eom_history,
        
        -- Months since last non-zero transaction (general activity)
        CASE 
            WHEN pm.monthly_total > 0 
            THEN 0
            ELSE ROW_NUMBER() OVER (
                PARTITION BY pm.dim_value, (pm.monthly_total > 0)
                ORDER BY pm.month DESC
            ) - 1
        END AS months_inactive,
        
        -- Properly track months since last EOM
        -- pm.months_since_last_eom,
        
        -- EOM pattern detection helpers
        -- Check if EOM occurs at regular intervals
        CASE 
            WHEN pm.total_nonzero_eom_count >= 3
                AND pm.months_of_history >= 12
            THEN 
                CASE
                    -- Monthly EOM pattern
                    WHEN pm.rolling_nonzero_eom_months >= 10 THEN 'MONTHLY'
                    -- Quarterly pattern
                    WHEN MOD(pm.total_nonzero_eom_count, 3) = 0 
                         AND pm.total_nonzero_eom_count >= FLOOR(pm.months_of_history / 4)
                    THEN 'QUARTERLY'
                    -- Semi-annual pattern
                    WHEN MOD(pm.total_nonzero_eom_count, 6) = 0
                         AND pm.total_nonzero_eom_count >= FLOOR(pm.months_of_history / 7)
                    THEN 'SEMIANNUAL'
                    -- Annual pattern
                    WHEN pm.total_nonzero_eom_count = FLOOR(pm.months_of_history / 12)
                    THEN 'ANNUAL'
                    ELSE 'IRREGULAR'
                END
            ELSE 'INSUFFICIENT_DATA'
        END AS eom_periodicity
        
    FROM 
        portfolio_metrics pm
)

-- =====================================================
-- STEP 6: DUAL IMPORTANCE CLASSIFICATION
-- =====================================================

, importance_classification AS (
    SELECT 
        ptm.*,
        
        -- Overall importance tier (using OVERALL-specific thresholds)
        CASE 
            WHEN ptm.cumulative_overall_portfolio_pct <= c.overall_critical_percentile
                OR ptm.rolling_avg_monthly_volume >= c.critical_monthly_avg_threshold
                OR ptm.rolling_max_transaction >= c.critical_max_transaction_threshold
            THEN 'CRITICAL'
            
            WHEN ptm.cumulative_overall_portfolio_pct <= c.overall_high_percentile
                OR ptm.rolling_avg_monthly_volume >= c.high_monthly_avg_threshold
                OR ptm.rolling_max_transaction >= c.high_max_transaction_threshold
            THEN 'HIGH'
            
            WHEN ptm.cumulative_overall_portfolio_pct <= c.overall_medium_percentile
                OR ptm.rolling_avg_monthly_volume >= c.medium_monthly_avg_threshold
                OR ptm.rolling_max_transaction >= c.medium_max_transaction_threshold
            THEN 'MEDIUM'
            
            ELSE 'LOW'
        END AS overall_importance_tier,
        
        -- EOM importance tier (using EOM-specific thresholds)
        CASE 
            WHEN ptm.cumulative_eom_portfolio_pct <= c.eom_critical_percentile
                OR ptm.rolling_avg_nonzero_eom >= c.critical_eom_monthly_threshold
                OR ptm.rolling_max_eom >= c.critical_max_eom_threshold
            THEN 'CRITICAL'
            
            WHEN ptm.cumulative_eom_portfolio_pct <= c.eom_high_percentile
                OR ptm.rolling_avg_nonzero_eom >= c.high_eom_monthly_threshold
                OR ptm.rolling_max_eom >= c.high_max_eom_threshold
            THEN 'HIGH'
            
            WHEN ptm.cumulative_eom_portfolio_pct <= c.eom_medium_percentile
                OR (ptm.rolling_avg_nonzero_eom >= c.medium_eom_monthly_threshold
                    AND ptm.total_nonzero_eom_count >= 3)
                OR ptm.rolling_max_eom >= c.medium_max_eom_threshold
            THEN 'MEDIUM'
            
            WHEN ptm.rolling_eom_volume_12m > 0
                OR ptm.total_nonzero_eom_count > 0
            THEN 'LOW'
            
            ELSE 'NONE'
        END AS eom_importance_tier,
        
        -- Overall importance score (0-1)
        CASE 
            WHEN ptm.total_portfolio_volume > 0 
            THEN ptm.rolling_total_volume_12m / ptm.total_portfolio_volume
            ELSE 0 
        END AS overall_importance_score,
        
        -- EOM importance score (0-1)
        CASE 
            WHEN ptm.total_portfolio_eom_volume > 0 
            THEN ptm.rolling_eom_volume_12m / ptm.total_portfolio_eom_volume
            ELSE 0 
        END AS eom_importance_score,
        
        -- EOM risk flag (high volume accounts with no EOM history)
        CASE 
            WHEN ptm.rolling_total_volume_12m >= c.eom_risk_volume_threshold
                AND ptm.has_eom_history = 0
                AND ptm.months_of_history >= c.eom_risk_min_months
            THEN 1
            ELSE 0
        END AS eom_risk_flag
        
    FROM 
        pattern_metrics ptm
    CROSS JOIN 
        config c
)

-- =====================================================
-- STEP 7: EOM PATTERN CLASSIFICATION (SMOOTH SCORING APPROACH)
-- =====================================================

, eom_smooth_scores AS (
    SELECT 
        ic.*,
        
        -- CONTINUOUS SMOOTH SCORES (no hard cutoffs)
        
        -- 1. REGULARITY SCORE: Sigmoid function for smooth transition
        -- Maps frequency [0,1] to score [0,100] with inflection at 0.5
        100 * (1 / (1 + EXP(-10 * (ic.eom_frequency - 0.5)))) AS regularity_score,
        
        -- 2. STABILITY SCORE: Inverse exponential decay based on CV
        -- High CV = low score, smoothly decreasing
        100 * EXP(-2 * GREATEST(ic.eom_cv, 0)) AS stability_score,
        
        -- 3. RECENCY SCORE: Exponential time decay based on months since last EOM
        CASE 
            WHEN ic.has_eom_history = 0 THEN 0
            WHEN ic.has_nonzero_eom = 1 THEN 100  -- Current month has EOM
            WHEN ic.months_since_last_eom = 1 THEN 80  -- Last month
            WHEN ic.months_since_last_eom = 2 THEN 64  -- 2 months ago
            WHEN ic.months_since_last_eom = 3 THEN 51  -- 3 months ago
            ELSE 100 * POWER(0.8, GREATEST(4, LEAST(24, ic.months_since_last_eom)))  -- Older
        END AS recency_score,
        
        -- 4. CONCENTRATION SCORE: How much of volume is at EOM
        -- Using logistic curve to smooth the transition
        100 * (1 / (1 + EXP(-5 * (ic.eom_concentration - 0.5)))) AS concentration_score,
        
        -- 5. VOLUME SCORE: Importance based on absolute volume (normalized)
        CASE 
            WHEN ic.total_portfolio_eom_volume > 0
            THEN 100 * (1 - EXP(-5 * ic.eom_importance_score))  -- Asymptotic growth
            ELSE 0
        END AS volume_importance_score
        
    FROM 
        importance_classification ic
)

-- Calculate distances to pattern archetypes in multi-dimensional space
, pattern_distances AS (
    SELECT 
        ess.*,
        
        -- Each pattern archetype is defined by ideal scores
        -- Calculate Euclidean distance to each archetype
        
        -- CONTINUOUS_STABLE: high regularity (90), high stability (80), medium recency (50)
        SQRT(
            POWER(90 - ess.regularity_score, 2) + 
            POWER(80 - ess.stability_score, 2) + 
            POWER(50 - ess.recency_score, 2) * 0.5  -- Lower weight on recency for continuous
        ) AS dist_continuous_stable,
        
        -- CONTINUOUS_VOLATILE: high regularity (90), low stability (20), medium recency (50)
        SQRT(
            POWER(90 - ess.regularity_score, 2) + 
            POWER(20 - ess.stability_score, 2) + 
            POWER(50 - ess.recency_score, 2) * 0.5
        ) AS dist_continuous_volatile,
        
        -- INTERMITTENT_ACTIVE: medium regularity (50), medium stability (50), high recency (90)
        SQRT(
            POWER(50 - ess.regularity_score, 2) + 
            POWER(50 - ess.stability_score, 2) + 
            POWER(90 - ess.recency_score, 2)
        ) AS dist_intermittent_active,
        
        -- INTERMITTENT_DORMANT: medium regularity (50), medium stability (50), low recency (20)
        SQRT(
            POWER(50 - ess.regularity_score, 2) + 
            POWER(50 - ess.stability_score, 2) + 
            POWER(20 - ess.recency_score, 2)
        ) AS dist_intermittent_dormant,
        
        -- RARE_RECENT: low regularity (15), any stability (50), high recency (85)
        SQRT(
            POWER(15 - ess.regularity_score, 2) + 
            POWER(50 - ess.stability_score, 2) * 0.3 +  -- Less weight on stability for rare
            POWER(85 - ess.recency_score, 2)
        ) AS dist_rare_recent,
        
        -- RARE_STALE: low regularity (15), any stability (50), low recency (15)
        SQRT(
            POWER(15 - ess.regularity_score, 2) + 
            POWER(50 - ess.stability_score, 2) * 0.3 +
            POWER(15 - ess.recency_score, 2)
        ) AS dist_rare_stale,
        
        -- NO_EOM: all zeros
        SQRT(
            POWER(0 - ess.regularity_score, 2) + 
            POWER(50 - ess.stability_score, 2) + 
            POWER(0 - ess.recency_score, 2)
        ) AS dist_no_eom,
        
        -- EMERGING: special case for new series
        CASE 
            WHEN ess.months_of_history <= 3 THEN 0
            ELSE 999
        END AS dist_emerging
        
    FROM 
        eom_smooth_scores ess
)

-- Convert distances to probabilities using softmax
, pattern_probabilities AS (
    SELECT 
        pd.*,
        
        -- Temperature parameter for softmax (lower = more confident assignments)
        20.0 AS temperature,
        
        -- Calculate softmax denominators for numerical stability
        (
            EXP(-pd.dist_continuous_stable / 20.0) +
            EXP(-pd.dist_continuous_volatile / 20.0) +
            EXP(-pd.dist_intermittent_active / 20.0) +
            EXP(-pd.dist_intermittent_dormant / 20.0) +
            EXP(-pd.dist_rare_recent / 20.0) +
            EXP(-pd.dist_rare_stale / 20.0) +
            CASE WHEN pd.has_eom_history = 0 THEN EXP(-pd.dist_no_eom / 20.0) ELSE 0 END +
            CASE WHEN pd.months_of_history <= 3 THEN EXP(-pd.dist_emerging / 20.0) ELSE 0 END
        ) AS softmax_denominator
        
    FROM 
        pattern_distances pd
)

-- Calculate final probabilities and classifications
, eom_pattern_classification AS (
    SELECT 
        pp.*,
        
        -- Probability memberships for each pattern
        CASE 
            WHEN pp.months_of_history <= 3 THEN 0
            ELSE EXP(-pp.dist_continuous_stable / pp.temperature) / pp.softmax_denominator
        END AS prob_continuous_stable,
        
        CASE 
            WHEN pp.months_of_history <= 3 THEN 0
            ELSE EXP(-pp.dist_continuous_volatile / pp.temperature) / pp.softmax_denominator
        END AS prob_continuous_volatile,
        
        CASE 
            WHEN pp.months_of_history <= 3 THEN 0
            ELSE EXP(-pp.dist_intermittent_active / pp.temperature) / pp.softmax_denominator
        END AS prob_intermittent_active,
        
        CASE 
            WHEN pp.months_of_history <= 3 THEN 0
            ELSE EXP(-pp.dist_intermittent_dormant / pp.temperature) / pp.softmax_denominator
        END AS prob_intermittent_dormant,
        
        CASE 
            WHEN pp.months_of_history <= 3 THEN 0
            ELSE EXP(-pp.dist_rare_recent / pp.temperature) / pp.softmax_denominator
        END AS prob_rare_recent,
        
        CASE 
            WHEN pp.months_of_history <= 3 THEN 0
            ELSE EXP(-pp.dist_rare_stale / pp.temperature) / pp.softmax_denominator
        END AS prob_rare_stale,
        
        CASE 
            WHEN pp.has_eom_history = 0 THEN EXP(-pp.dist_no_eom / pp.temperature) / pp.softmax_denominator
            ELSE 0
        END AS prob_no_eom,
        
        CASE 
            WHEN pp.months_of_history <= 3 THEN EXP(-pp.dist_emerging / pp.temperature) / pp.softmax_denominator
            ELSE 0
        END AS prob_emerging,
        
        -- Primary classification (highest probability)
        CASE 
            WHEN pp.months_of_history <= 3 THEN 'EMERGING'
            WHEN pp.has_eom_history = 0 AND pp.months_of_history >= 6 THEN 'NO_EOM'
            ELSE
                CASE 
                    GREATEST(
                        EXP(-pp.dist_continuous_stable / pp.temperature),
                        EXP(-pp.dist_continuous_volatile / pp.temperature),
                        EXP(-pp.dist_intermittent_active / pp.temperature),
                        EXP(-pp.dist_intermittent_dormant / pp.temperature),
                        EXP(-pp.dist_rare_recent / pp.temperature),
                        EXP(-pp.dist_rare_stale / pp.temperature)
                    )
                    WHEN EXP(-pp.dist_continuous_stable / pp.temperature) THEN 'CONTINUOUS_STABLE'
                    WHEN EXP(-pp.dist_continuous_volatile / pp.temperature) THEN 'CONTINUOUS_VOLATILE'
                    WHEN EXP(-pp.dist_intermittent_active / pp.temperature) THEN 'INTERMITTENT_ACTIVE'
                    WHEN EXP(-pp.dist_intermittent_dormant / pp.temperature) THEN 'INTERMITTENT_DORMANT'
                    WHEN EXP(-pp.dist_rare_recent / pp.temperature) THEN 'RARE_RECENT'
                    WHEN EXP(-pp.dist_rare_stale / pp.temperature) THEN 'RARE_STALE'
                END
        END AS eom_pattern,
        
        -- Pattern confidence (highest probability value)
        GREATEST(
            CASE WHEN pp.months_of_history <= 3 THEN 1.0 
                 WHEN pp.has_eom_history = 0 THEN 1.0 
                 ELSE 0 END,
            EXP(-pp.dist_continuous_stable / pp.temperature) / pp.softmax_denominator,
            EXP(-pp.dist_continuous_volatile / pp.temperature) / pp.softmax_denominator,
            EXP(-pp.dist_intermittent_active / pp.temperature) / pp.softmax_denominator,
            EXP(-pp.dist_intermittent_dormant / pp.temperature) / pp.softmax_denominator,
            EXP(-pp.dist_rare_recent / pp.temperature) / pp.softmax_denominator,
            EXP(-pp.dist_rare_stale / pp.temperature) / pp.softmax_denominator
        ) AS eom_pattern_confidence,
        
        -- Classification uncertainty (entropy-based)
        -- Higher entropy = more uncertain
        -(
            COALESCE(EXP(-pp.dist_continuous_stable / pp.temperature) / pp.softmax_denominator * 
                     LN(NULLIF(EXP(-pp.dist_continuous_stable / pp.temperature) / pp.softmax_denominator, 0)), 0) +
            COALESCE(EXP(-pp.dist_continuous_volatile / pp.temperature) / pp.softmax_denominator * 
                     LN(NULLIF(EXP(-pp.dist_continuous_volatile / pp.temperature) / pp.softmax_denominator, 0)), 0) +
            COALESCE(EXP(-pp.dist_intermittent_active / pp.temperature) / pp.softmax_denominator * 
                     LN(NULLIF(EXP(-pp.dist_intermittent_active / pp.temperature) / pp.softmax_denominator, 0)), 0) +
            COALESCE(EXP(-pp.dist_intermittent_dormant / pp.temperature) / pp.softmax_denominator * 
                     LN(NULLIF(EXP(-pp.dist_intermittent_dormant / pp.temperature) / pp.softmax_denominator, 0)), 0) +
            COALESCE(EXP(-pp.dist_rare_recent / pp.temperature) / pp.softmax_denominator * 
                     LN(NULLIF(EXP(-pp.dist_rare_recent / pp.temperature) / pp.softmax_denominator, 0)), 0) +
            COALESCE(EXP(-pp.dist_rare_stale / pp.temperature) / pp.softmax_denominator * 
                     LN(NULLIF(EXP(-pp.dist_rare_stale / pp.temperature) / pp.softmax_denominator, 0)), 0)
        ) AS classification_entropy,
        
        -- EOM risk flag for high-value volatile accounts
        CASE 
            WHEN pp.eom_importance_tier IN ('CRITICAL', 'HIGH')
                AND pp.stability_score < 30  -- Low stability
                AND pp.concentration_score >= 50  -- Meaningful EOM portion
            THEN 1
            ELSE 0
        END AS eom_high_risk_flag
        
    FROM 
        pattern_probabilities pp
)

-- =====================================================
-- STEP 8: GENERAL TIMESERIES PATTERN CLASSIFICATION
-- =====================================================

, general_pattern_classification AS (
    SELECT 
        epc.*,
        
        -- General timeseries behavioral patterns (independent of EOM)
        CASE 
            -- INACTIVE: No recent activity
            WHEN epc.months_inactive >= c.inactive_months
            THEN 'INACTIVE'
            
            -- EMERGING: Too new to classify
            WHEN epc.months_of_history <= c.emerging_months
            THEN 'EMERGING'
            
            -- HIGHLY_SEASONAL: Strong quarterly/yearly patterns
            WHEN epc.quarter_end_concentration >= c.ts_seasonal_concentration_threshold
                OR epc.year_end_concentration >= 0.5
            THEN 'HIGHLY_SEASONAL'
            
            -- INTERMITTENT: Sporadic overall activity
            WHEN epc.activity_rate <= c.ts_intermittent_threshold
                OR epc.active_months_12m <= 4
            THEN 'INTERMITTENT'
            
            -- VOLATILE: Regular but unpredictable amounts
            WHEN epc.monthly_cv >= c.ts_high_volatility_threshold
                AND epc.transaction_regularity >= c.ts_medium_regularity_threshold
            THEN 'VOLATILE'
            
            -- MODERATELY_VOLATILE: Some variability
            WHEN epc.monthly_cv >= c.ts_medium_volatility_threshold
                AND epc.transaction_regularity >= c.ts_medium_regularity_threshold
            THEN 'MODERATELY_VOLATILE'
            
            -- STABLE: Regular and predictable
            WHEN epc.monthly_cv < c.ts_medium_volatility_threshold
                AND epc.transaction_regularity >= c.ts_high_regularity_threshold
            THEN 'STABLE'
            
            -- CONCENTRATED: Activity clustered in time
            WHEN epc.transaction_dispersion < 5  -- Low day dispersion
                AND epc.transaction_regularity >= c.ts_medium_regularity_threshold
            THEN 'CONCENTRATED'
            
            -- DISTRIBUTED: Activity spread throughout month
            WHEN epc.transaction_dispersion >= 8  -- High day dispersion
                AND epc.transaction_regularity >= c.ts_medium_regularity_threshold
            THEN 'DISTRIBUTED'
            
            ELSE 'MIXED'
            
        END AS general_pattern
        
    FROM 
        eom_pattern_classification epc
    CROSS JOIN 
        config c
)

-- =====================================================
-- STEP 9: COMBINED PRIORITY AND RECOMMENDATIONS
-- =====================================================

, final_classification AS (
    SELECT 
        gpc.*,
        
        -- Combined priority (1-10 scale) considering both patterns
        CASE 
            -- Highest priority: Critical importance with complex patterns
            WHEN (gpc.overall_importance_tier = 'CRITICAL' OR gpc.eom_importance_tier = 'CRITICAL')
                AND (gpc.eom_pattern IN ('PURE_EOM_VOLATILE', 'INTERMITTENT_EOM') 
                     OR gpc.general_pattern IN ('VOLATILE', 'INTERMITTENT'))
            THEN 10
            
            WHEN (gpc.overall_importance_tier = 'CRITICAL' OR gpc.eom_importance_tier = 'CRITICAL')
                AND (gpc.eom_pattern = 'SEASONAL_EOM' OR gpc.general_pattern = 'HIGHLY_SEASONAL')
            THEN 9
            
            WHEN (gpc.overall_importance_tier = 'CRITICAL' OR gpc.eom_importance_tier = 'CRITICAL')
            THEN 8
            
            -- High importance with complex patterns
            WHEN (gpc.overall_importance_tier = 'HIGH' OR gpc.eom_importance_tier = 'HIGH')
                AND (gpc.eom_pattern IN ('PURE_EOM_VOLATILE', 'INTERMITTENT_EOM') 
                     OR gpc.general_pattern IN ('VOLATILE', 'INTERMITTENT'))
            THEN 7
            
            WHEN (gpc.overall_importance_tier = 'HIGH' OR gpc.eom_importance_tier = 'HIGH')
            THEN 6
            
            -- EOM risk cases or medium importance with complex patterns
            WHEN gpc.eom_risk_flag = 1
                OR (gpc.overall_importance_tier = 'MEDIUM' 
                    AND gpc.general_pattern IN ('VOLATILE', 'INTERMITTENT'))
            THEN 5
            
            -- Medium importance
            WHEN gpc.overall_importance_tier = 'MEDIUM' OR gpc.eom_importance_tier = 'MEDIUM'
            THEN 4
            
            -- Low importance but active
            WHEN gpc.general_pattern NOT IN ('INACTIVE', 'EMERGING')
            THEN 3
            
            -- Emerging patterns
            WHEN gpc.general_pattern = 'EMERGING' OR gpc.eom_pattern = 'EMERGING_EOM'
            THEN 2
            
            -- Inactive
            ELSE 1
            
        END AS combined_priority,
        
        -- Recommended forecasting method based on BOTH patterns
        CASE 
            -- No EOM cases
            WHEN gpc.eom_pattern = 'NO_EOM' 
            THEN 'Zero_EOM_Forecast'
            
            -- Pure EOM patterns
            WHEN gpc.eom_pattern IN ('PURE_EOM_STABLE', 'PURE_EOM_VOLATILE') THEN
                CASE 
                    WHEN gpc.eom_importance_tier IN ('CRITICAL', 'HIGH')
                        AND gpc.eom_pattern = 'PURE_EOM_VOLATILE'
                    THEN 'XGBoost_EOM_Focus'
                    WHEN gpc.eom_importance_tier IN ('CRITICAL', 'HIGH')
                    THEN 'SARIMA_EOM'
                    ELSE 'Simple_MA_EOM'
                END
            
            -- Regular EOM with various general patterns
            WHEN gpc.eom_pattern = 'REGULAR_EOM' THEN
                CASE 
                    WHEN gpc.general_pattern IN ('VOLATILE', 'MODERATELY_VOLATILE')
                    THEN 'Hybrid_Model'
                    WHEN gpc.general_pattern = 'STABLE'
                    THEN 'Auto_ARIMA'
                    ELSE 'Weighted_Average'
                END
            
            -- Intermittent patterns
            WHEN gpc.eom_pattern = 'INTERMITTENT_EOM' OR gpc.general_pattern = 'INTERMITTENT' THEN
                CASE 
                    WHEN gpc.overall_importance_tier IN ('CRITICAL', 'HIGH')
                    THEN 'Croston_Method'
                    ELSE 'Zero_Inflated_Model'
                END
            
            -- Seasonal patterns
            WHEN gpc.eom_pattern = 'SEASONAL_EOM' OR gpc.general_pattern = 'HIGHLY_SEASONAL' THEN
                CASE 
                    WHEN gpc.overall_importance_tier IN ('CRITICAL', 'HIGH')
                    THEN 'Seasonal_Decomposition'
                    ELSE 'Seasonal_Naive'
                END
            
            -- Mixed or rare EOM
            WHEN gpc.eom_pattern IN ('MIXED_EOM', 'RARE_EOM') THEN
                CASE 
                    WHEN gpc.general_pattern IN ('VOLATILE', 'MODERATELY_VOLATILE')
                    THEN 'Ensemble_Method'
                    WHEN gpc.general_pattern = 'STABLE'
                    THEN 'Prophet'
                    ELSE 'Historical_Average'
                END
            
            -- General patterns without significant EOM
            WHEN gpc.general_pattern = 'VOLATILE' THEN 'XGBoost_Full_Series'
            WHEN gpc.general_pattern = 'STABLE' THEN 'Linear_Trend'
            WHEN gpc.general_pattern = 'CONCENTRATED' THEN 'Peak_Detection_Model'
            WHEN gpc.general_pattern = 'DISTRIBUTED' THEN 'Daily_Decomposition'
            
            -- Default cases
            WHEN gpc.general_pattern = 'INACTIVE' THEN 'Zero_Forecast'
            WHEN gpc.general_pattern = 'EMERGING' THEN 'Conservative_MA'
            
            ELSE 'Historical_Average'
            
        END AS recommended_method,
        
        -- Forecast complexity (1-5)
        CASE 
            WHEN (gpc.eom_pattern IN ('PURE_EOM_VOLATILE', 'INTERMITTENT_EOM')
                  OR gpc.general_pattern IN ('VOLATILE', 'INTERMITTENT'))
                AND (gpc.overall_importance_tier IN ('CRITICAL', 'HIGH') 
                     OR gpc.eom_importance_tier IN ('CRITICAL', 'HIGH'))
            THEN 5
            
            WHEN (gpc.eom_pattern = 'SEASONAL_EOM' OR gpc.general_pattern = 'HIGHLY_SEASONAL')
                AND (gpc.overall_importance_tier IN ('CRITICAL', 'HIGH'))
            THEN 4
            
            WHEN gpc.eom_pattern IN ('PURE_EOM_STABLE', 'REGULAR_EOM')
                OR gpc.general_pattern = 'STABLE'
                OR gpc.overall_importance_tier IN ('CRITICAL', 'HIGH')
            THEN 3
            
            WHEN gpc.overall_importance_tier = 'MEDIUM'
                OR gpc.eom_importance_tier = 'MEDIUM'
            THEN 2
            
            ELSE 1
        END AS forecast_complexity,
        
        -- Combined segment name showing both patterns
        CONCAT(
            gpc.overall_importance_tier,
            '_',
            gpc.general_pattern,
            '__',
            gpc.eom_importance_tier,
            'EOM_',
            gpc.eom_pattern
        ) AS full_segment_name,
        
        -- Simplified segment name for reporting
        CONCAT(
            CASE 
                WHEN gpc.overall_importance_tier = gpc.eom_importance_tier 
                THEN gpc.overall_importance_tier
                ELSE CONCAT(gpc.overall_importance_tier, '/', gpc.eom_importance_tier, 'EOM')
            END,
            '_',
            gpc.general_pattern,
            '_',
            gpc.eom_pattern
        ) AS segment_name
        
    FROM 
        general_pattern_classification gpc
)

-- =====================================================
-- STEP 10: CALCULATE GROWTH METRICS
-- =====================================================

, with_growth AS (
    SELECT 
        fc.*,
        
        -- Growth rates
        CASE 
            WHEN fc.eom_amount_12m_ago > 0 
            THEN (fc.eom_amount - fc.eom_amount_12m_ago) / fc.eom_amount_12m_ago
            WHEN fc.eom_amount_12m_ago = 0 AND fc.eom_amount > 0
            THEN 999.99
            ELSE 0
        END AS eom_yoy_growth,
        
        -- Month-over-month growth
        CASE 
            WHEN LAG(fc.eom_amount, 1) OVER (PARTITION BY fc.dim_value ORDER BY fc.month) > 0
            THEN (fc.eom_amount - LAG(fc.eom_amount, 1) OVER (PARTITION BY fc.dim_value ORDER BY fc.month)) 
                 / LAG(fc.eom_amount, 1) OVER (PARTITION BY fc.dim_value ORDER BY fc.month)
            ELSE 0 
        END AS eom_mom_growth
        
    FROM 
        final_classification fc
)

-- =====================================================
-- FINAL OUTPUT
-- =====================================================

SELECT 
    wg.dim_value,
    wg.month AS forecast_month,
    wg.year,
    wg.month_num,
    
    -- Target variable
    wg.eom_amount AS target_eom_amount,
    
    -- Dual importance metrics
    wg.overall_importance_tier,
    wg.eom_importance_tier,
    ROUND(wg.overall_importance_score, 5) AS overall_importance_score,
    ROUND(wg.eom_importance_score, 5) AS eom_importance_score,
    wg.eom_risk_flag,
    wg.has_eom_history,
    
    -- SMOOTH SCORES (new)
    ROUND(wg.regularity_score, 1) AS eom_regularity_score,
    ROUND(wg.stability_score, 1) AS eom_stability_score,
    ROUND(wg.recency_score, 1) AS eom_recency_score,
    ROUND(wg.concentration_score, 1) AS eom_concentration_score,
    ROUND(wg.volume_importance_score, 1) AS eom_volume_score,
    
    -- PATTERN PROBABILITIES (new - shows smooth transitions)
    wg.eom_pattern AS eom_pattern_primary,
    ROUND(wg.eom_pattern_confidence * 100, 1) AS eom_pattern_confidence_pct,
    ROUND(wg.prob_continuous_stable * 100, 1) AS prob_continuous_stable_pct,
    ROUND(wg.prob_continuous_volatile * 100, 1) AS prob_continuous_volatile_pct,
    ROUND(wg.prob_intermittent_active * 100, 1) AS prob_intermittent_active_pct,
    ROUND(wg.prob_intermittent_dormant * 100, 1) AS prob_intermittent_dormant_pct,
    ROUND(wg.prob_rare_recent * 100, 1) AS prob_rare_recent_pct,
    ROUND(wg.prob_rare_stale * 100, 1) AS prob_rare_stale_pct,
    ROUND(wg.classification_entropy, 3) AS pattern_uncertainty,
    
    -- General timeseries pattern
    wg.general_pattern,
    
    -- Combined metrics
    wg.segment_name,
    wg.full_segment_name,
    wg.combined_priority,
    wg.recommended_method,
    wg.forecast_complexity,
    
    -- Volume metrics
    wg.rolling_total_volume_12m AS total_volume_12m,
    wg.rolling_eom_volume_12m AS eom_volume_12m,
    wg.rolling_non_eom_volume_12m AS non_eom_volume_12m,
    wg.rolling_avg_monthly_volume AS avg_monthly_volume,
    wg.rolling_max_transaction AS max_transaction,
    wg.rolling_max_eom AS max_eom_transaction,
    
    -- Raw EOM metrics (for validation)
    ROUND(wg.eom_concentration, 3) AS eom_concentration,
    ROUND(wg.eom_predictability, 3) AS eom_predictability,
    ROUND(wg.eom_frequency, 3) AS eom_frequency,
    ROUND(wg.eom_zero_ratio, 3) AS eom_zero_ratio,
    ROUND(wg.eom_cv, 3) AS eom_cv,
    
    -- General timeseries pattern features
    ROUND(wg.monthly_cv, 3) AS monthly_cv,
    ROUND(wg.transaction_regularity, 3) AS transaction_regularity,
    ROUND(wg.activity_rate, 3) AS activity_rate,
    ROUND(wg.transaction_dispersion, 2) AS transaction_dispersion,
    ROUND(wg.quarter_end_concentration, 3) AS quarter_end_concentration,
    ROUND(wg.year_end_concentration, 3) AS year_end_concentration,
    
    -- Activity indicators
    wg.active_months_12m,
    wg.total_nonzero_eom_count,
    wg.months_inactive,
    wg.months_of_history,
    
    -- Portfolio percentiles (showing both)
    ROUND(wg.cumulative_overall_portfolio_pct, 4) AS cumulative_overall_portfolio_pct,
    ROUND(wg.cumulative_eom_portfolio_pct, 4) AS cumulative_eom_portfolio_pct,
    
    -- Lagged values for forecasting
    COALESCE(wg.eom_amount_1m_ago, 0) AS lag_1m_eom,
    COALESCE(wg.eom_amount_3m_ago, 0) AS lag_3m_eom,
    COALESCE(wg.eom_amount_12m_ago, 0) AS lag_12m_eom,
    ROUND(wg.eom_ma3, 2) AS eom_ma3,
    
    -- Growth metrics
    CASE WHEN wg.months_of_history >= 12 THEN ROUND(wg.eom_yoy_growth, 3) END AS eom_yoy_growth,
    ROUND(wg.eom_mom_growth, 3) AS eom_mom_growth,
    
    -- Calendar features
    wg.is_quarter_end,
    wg.is_year_end,
    wg.month_num AS month_of_year,
    
    -- Current month status
    CASE WHEN wg.eom_amount = 0 THEN 1 ELSE 0 END AS is_zero_eom,
    wg.has_nonzero_eom AS current_month_has_eom,

    -- raw rolling features
    wg.active_months_12m AS raw_rf__active_months_12m,
    wg.rolling_avg_day_dispersion AS raw_rf__rolling_avg_day_dispersion,
    wg.rolling_avg_monthly_volume AS raw_rf__rolling_avg_monthly_volume,
    wg.rolling_avg_non_eom AS raw_rf__rolling_avg_non_eom,
    wg.rolling_avg_nonzero_eom AS raw_rf__rolling_avg_nonzero_eom,
    wg.rolling_avg_transactions AS raw_rf__rolling_avg_transactions,
    wg.rolling_eom_volume_12m AS raw_rf__rolling_eom_volume_12m,
    wg.rolling_max_eom AS raw_rf__rolling_max_eom,
    wg.rolling_max_transaction AS raw_rf__rolling_max_transaction,
    wg.rolling_non_eom_volume_12m AS raw_rf__rolling_non_eom_volume_12m,
    wg.rolling_nonzero_eom_months AS raw_rf__rolling_nonzero_eom_months,
    wg.rolling_quarter_end_volume AS raw_rf__rolling_quarter_end_volume,
    wg.rolling_std_eom AS raw_rf__rolling_std_eom,
    wg.rolling_std_monthly AS raw_rf__rolling_std_monthly,
    wg.rolling_std_transactions AS raw_rf__rolling_std_transactions,
    wg.rolling_total_volume_12m AS raw_rf__rolling_total_volume_12m,
    wg.rolling_year_end_volume AS raw_rf__rolling_year_end_volume,
    wg.rolling_zero_eom_months AS raw_rf__rolling_zero_eom_months,
    wg.total_nonzero_eom_count AS raw_rf__total_nonzero_eom_count,
    wg.eom_amount_12m_ago AS raw_rf__eom_amount_12m_ago,
    wg.eom_amount_3m_ago AS raw_rf__eom_amount_3m_ago,
    wg.eom_amount_1m_ago AS raw_rf__eom_amount_1m_ago,
    wg.eom_ma3 AS raw_rf__eom_ma3,
    wg.months_of_history AS raw_rf__months_of_history,
    wg.months_since_last_eom AS raw_rf__months_since_last_eom,

    -- raw pattern metrics
    wg.eom_concentration AS raw_pm__eom_concentration,
    wg.eom_predictability AS raw_pm__eom_predictability,  
    wg.eom_frequency AS raw_pm__eom_frequency,
    wg.eom_zero_ratio AS raw_pm__eom_zero_ratio, 
    wg.eom_spike_ratio AS raw_pm__eom_spike_ratio,
    wg.eom_cv AS raw_pm__eom_cv,
    wg.monthly_cv AS raw_pm__monthly_cv,
    wg.transaction_regularity AS raw_pm__transaction_regularity,  
    wg.activity_rate AS raw_pm__activity_rate,
    wg.quarter_end_concentration AS raw_pm__quarter_end_concentration,
    wg.year_end_concentration AS raw_pm__year_end_concentration,
    wg.transaction_dispersion AS raw_pm__transaction_dispersion,
    wg.has_eom_history AS raw_pm__has_eom_history,
    wg.months_inactive AS raw_pm__months_inactive,
    wg.eom_periodicity AS raw_pm__eom_periodicity
    
FROM 
    with_growth wg
WHERE 
    forecast_month = '2025-07-01'  -- Filter for specific month if needed
    -- AND wg.eom_importance_tier NOT IN ('LOW', 'NONE')  -- Optional filter
ORDER BY 
    wg.combined_priority DESC,
    wg.dim_value, 
    wg.month
"""
BAN6800 Capstone — Stanbic IBTC Credit Risk Intelligence Platform
Bias Detection Suite
File: src/features/bias_detection.py
Commit to: github.com/Layorr/stanbic-credit-risk

References:
  - Mehrabi et al. (2021). A survey on bias and fairness in ML.
    ACM Computing Surveys, 54(6). https://doi.org/10.1145/3457607
  - Bird et al. (2020). Fairlearn. https://fairlearn.org
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def check_representation_bias(df: pd.DataFrame,
                               target: str,
                               protected_col: str) -> Dict[str, Any]:
    """
    Check for representation bias in training data by comparing
    default rates across subgroups of a protected characteristic.

    Args:
        df: Training dataframe
        target: Name of target column (0=repaid, 1=default)
        protected_col: Column representing protected characteristic

    Returns:
        Dictionary of bias metrics
    """
    if protected_col not in df.columns:
        logger.warning(f"Protected column '{protected_col}' not found. Skipping.")
        return {'status': 'skipped', 'reason': 'column_not_found'}

    groups = df.groupby(protected_col)[target].agg(['mean', 'count'])
    groups.columns = ['default_rate', 'count']
    groups['proportion'] = groups['count'] / len(df)

    max_rate = groups['default_rate'].max()
    min_rate = groups['default_rate'].min()
    disparity = max_rate - min_rate
    disparate_impact = min_rate / max_rate if max_rate > 0 else 1.0

    result = {
        'protected_column': protected_col,
        'subgroup_stats': groups.to_dict(),
        'max_default_rate': round(max_rate, 4),
        'min_default_rate': round(min_rate, 4),
        'max_disparity': round(disparity, 4),
        'disparate_impact_ratio': round(disparate_impact, 4),
        'passes_5pct_threshold': disparity <= 0.05,
        'passes_80pct_rule': disparate_impact >= 0.80,
    }

    if disparity > 0.05:
        logger.warning(f"BIAS WARNING: {protected_col} disparity = {disparity:.3f} > 5% threshold")
    else:
        logger.info(f"BIAS CHECK PASS: {protected_col} disparity = {disparity:.3f} ≤ 5%")

    return result


def check_missing_value_bias(df: pd.DataFrame,
                              feature: str,
                              protected_col: str) -> Dict[str, Any]:
    """
    Check whether missing values in a feature are distributed
    disproportionately across protected groups (Mehrabi et al., 2021).
    """
    if feature not in df.columns or protected_col not in df.columns:
        return {'status': 'skipped'}

    missing_by_group = df.groupby(protected_col)[feature].apply(
        lambda x: x.isnull().mean()
    ).round(4)

    max_missing = missing_by_group.max()
    min_missing = missing_by_group.min()
    disparity = max_missing - min_missing

    result = {
        'feature': feature,
        'protected_column': protected_col,
        'missing_by_group': missing_by_group.to_dict(),
        'max_missing_rate': max_missing,
        'min_missing_rate': min_missing,
        'disparity': round(disparity, 4),
        'flagged': disparity > 0.10,
    }

    if disparity > 0.10:
        logger.warning(f"MISSING BIAS: {feature} missing rate disparity across {protected_col} = {disparity:.3f}")

    return result


def run_bias_checks(df: pd.DataFrame,
                    target: str = 'TARGET',
                    protected: str = 'CODE_GENDER') -> Dict[str, Any]:
    """
    Run full bias detection suite and return consolidated report.
    Consistent with Module 1 Fairness Objectives (≤5% disparity threshold).
    """
    logger.info("Running bias detection suite...")
    report = {}

    # 1. Representation bias by default rate
    report['default_rate_bias'] = check_representation_bias(df, target, protected)

    # 2. Missing value bias for key bureau feature
    report['ext_source_1_bias'] = check_missing_value_bias(df, 'EXT_SOURCE_1', protected)
    report['ext_source_2_bias'] = check_missing_value_bias(df, 'EXT_SOURCE_2', protected)

    # 3. Class distribution check
    class_dist = df[target].value_counts(normalize=True).to_dict()
    report['class_distribution'] = {
        'default_rate': round(class_dist.get(1, 0), 4),
        'repaid_rate': round(class_dist.get(0, 0), 4),
        'imbalance_ratio': round(class_dist.get(0, 0) / class_dist.get(1, 1), 2),
    }

    # 4. Overall pass/fail
    max_disp = report['default_rate_bias'].get('max_disparity', 0)
    report['max_disparity'] = max_disp
    report['overall_pass'] = max_disp <= 0.05
    report['summary'] = f"{'PASS' if report['overall_pass'] else 'WARN'}: max disparity = {max_disp:.3f}"

    logger.info(f"Bias detection complete: {report['summary']}")
    return report


if __name__ == '__main__':
    # Example usage with synthetic data for testing
    import numpy as np
    np.random.seed(42)
    n = 1000
    test_df = pd.DataFrame({
        'SK_ID_CURR': range(n),
        'TARGET': np.random.binomial(1, 0.08, n),
        'CODE_GENDER': np.random.choice(['M', 'F'], n),
        'EXT_SOURCE_1': np.where(np.random.random(n) < 0.56, np.nan,
                                  np.random.uniform(0, 1, n)),
        'EXT_SOURCE_2': np.random.uniform(0, 1, n),
    })

    report = run_bias_checks(test_df, target='TARGET', protected='CODE_GENDER')
    print("\n=== BIAS DETECTION REPORT ===")
    print(f"Summary: {report['summary']}")
    print(f"Overall pass: {report['overall_pass']}")
    print(f"Class distribution: {report['class_distribution']}")

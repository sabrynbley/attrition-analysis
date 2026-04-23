import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "employee_id":     [1,       2,       3,    4,    5,    6,    7,    8],
            "department":      ["Sales", "Sales", "HR", "HR", "HR", "IT", "IT", "IT"],
            "monthly_income":  [4000,    6000,    5000, 7000, 5500, 4500, 8000, 6500],
            "job_satisfaction":[1,       3,       2,    4,    2,    1,    3,    4],
            "overtime":        ["Yes",   "No",    "Yes","No", "No", "Yes","No", "No"],
            "attrition":       ["Yes",   "No",    "Yes","No", "No", "Yes","No", "No"],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_rate_all_leavers(sample_df):
    all_yes = sample_df.copy()
    all_yes["attrition"] = "Yes"
    assert attrition_rate(all_yes) == 100.0


def test_attrition_rate_no_leavers(sample_df):
    all_no = sample_df.copy()
    all_no["attrition"] = "No"
    assert attrition_rate(all_no) == 0.0


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_rates(sample_df):
    result = attrition_by_department(sample_df)
    rates = dict(zip(result["department"], result["attrition_rate"]))
    assert rates["Sales"] == 50.0   # 1 leaver out of 2
    assert rates["HR"] == 33.33     # 1 leaver out of 3
    assert rates["IT"] == 33.33     # 1 leaver out of 3


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_rates(sample_df):
    result = attrition_by_overtime(sample_df)
    rates = dict(zip(result["overtime"], result["attrition_rate"]))
    assert rates["Yes"] == 100.0    # all 3 overtime workers left
    assert rates["No"] == 0.0       # no non-overtime workers left


# --- average_income_by_attrition ---

def test_average_income_by_attrition(sample_df):
    result = average_income_by_attrition(sample_df)
    income = dict(zip(result["attrition"], result["avg_monthly_income"]))
    assert income["Yes"] == 4500.0  # (4000 + 5000 + 4500) / 3
    assert income["No"] == 6600.0   # (6000 + 7000 + 5500 + 8000 + 6500) / 5


# --- satisfaction_summary ---

def test_satisfaction_summary_rates(sample_df):
    # Divides leavers by group size, not by total leavers across the dataset.
    result = satisfaction_summary(sample_df)
    rates = dict(zip(result["job_satisfaction"], result["attrition_rate"]))
    assert rates[1] == 100.0    # 2 leavers out of 2 employees
    assert rates[2] == 50.0     # 1 leaver out of 2 employees
    assert rates[3] == 0.0
    assert rates[4] == 0.0


def test_satisfaction_summary_sorted_ascending(sample_df):
    result = satisfaction_summary(sample_df)
    levels = list(result["job_satisfaction"])
    assert levels == sorted(levels)

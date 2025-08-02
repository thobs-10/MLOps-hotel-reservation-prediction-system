"""
Robust data validation module for hotel reservation ML pipeline.

This module provides comprehensive validation for cleaned data after ingestion,
including schema validation, data quality checks, business logic validation,
and feature engineering readiness checks.
"""

from typing import Any, Dict, List

import pandas as pd
import pandera as pa
from loguru import logger
from pandera import Check, Column, DataFrameSchema
from zenml.steps import step

from src.entity.constants import FeatureEngineeringConstants


def _try_read_method(data: Any) -> pd.DataFrame:
    """Try to unwrap data using the read() method."""
    if hasattr(data, "read") and callable(data.read):
        result = data.read()
        if isinstance(result, pd.DataFrame):
            return result
    raise ValueError("Cannot unwrap using read() method")


def _try_value_attribute(data: Any) -> pd.DataFrame:
    """Try to unwrap data using the .value attribute."""
    if hasattr(data, "value"):
        if isinstance(data.value, pd.DataFrame):
            return data.value
        elif hasattr(data.value, "read") and callable(data.value.read):
            result = data.value.read()
            if isinstance(result, pd.DataFrame):
                return result
    raise ValueError("Cannot unwrap using value attribute")


def _try_callable_artifact(data: Any) -> pd.DataFrame:
    """Try to unwrap data by calling it directly."""
    if callable(data):
        result = data()
        if isinstance(result, pd.DataFrame):
            return result
    raise ValueError("Cannot unwrap by calling artifact")


def _unwrap_zenml_artifact(data: Any) -> pd.DataFrame:
    """
    Utility function to unwrap ZenML artifacts and return the actual DataFrame.

    Args:
        data: Potentially wrapped DataFrame from ZenML

    Returns:
        pd.DataFrame: The actual DataFrame
    """
    # If it's already a DataFrame, return as-is
    if isinstance(data, pd.DataFrame):
        return data

    # Handle ZenML StepArtifact - this is the actual type that wraps the DataFrame
    if hasattr(data, "__class__") and "StepArtifact" in str(data.__class__):
        # Try different unwrapping strategies
        for unwrap_func in [
            _try_read_method,
            _try_value_attribute,
            _try_callable_artifact,
        ]:
            try:
                return unwrap_func(data)
            except ValueError:
                continue

    # If we can't unwrap it, raise a descriptive error
    raise TypeError(
        f"Cannot unwrap data of type {type(data)} to DataFrame. "
        f"This appears to be a ZenML StepArtifact. "
        f"Try calling the function that returns this artifact directly within a ZenML pipeline context, "
        f"or ensure the step has proper return type annotations."
    )


class DataValidationError(Exception):
    """Custom exception for data validation errors."""

    pass


class ValidationResult:
    """Container for validation results with detailed reporting."""

    def __init__(self, is_valid: bool, errors=None, warnings=None):
        self.is_valid = is_valid
        self.errors = errors if errors is not None else []
        self.warnings = warnings if warnings is not None else []
        self.report = self._generate_report()

    def _generate_report(self) -> str:
        """Generate a formatted validation report."""
        status = "✅ PASSED" if self.is_valid else "❌ FAILED"
        report = f"\n📋 VALIDATION REPORT - {status}\n" + "=" * 50 + "\n"

        if self.errors:
            report += "\n🚨 ERRORS:\n"
            for error in self.errors:
                report += f"  - {error}\n"

        if self.warnings:
            report += "\n⚠️ WARNINGS:\n"
            for warning in self.warnings:
                report += f"  - {warning}\n"

        if not self.errors and not self.warnings:
            report += "\n🎉 All validation checks passed successfully!\n"

        return report


class HotelDataValidator:
    """Comprehensive validator for hotel reservation data."""

    def __init__(self):
        self.fe_constants = FeatureEngineeringConstants()
        self._schema = None

    @property
    def schema(self) -> DataFrameSchema:
        """Lazily create and cache the validation schema."""
        if self._schema is None:
            self._schema = self._create_comprehensive_schema()
        return self._schema

    def _create_comprehensive_schema(self) -> DataFrameSchema:
        """Create a comprehensive validation schema for cleaned data."""
        schema_columns = {}

        # Add categorical columns (should be 'category' dtype after data ingestion)
        for col in self.fe_constants.categorical_columns:
            schema_columns[col] = Column(
                pa.Category,
                nullable=False,
                description=f"Categorical column: {col}",
            )

        # Add numerical columns with domain-specific validations
        for col in self.fe_constants.numerical_columns:
            schema_columns[col] = self._get_numerical_column_schema(col)

        # Add target column
        schema_columns[self.fe_constants.target_column] = Column(
            pa.String,
            Check.isin(["Canceled", "Not_Canceled"], error="Invalid booking status"),
            nullable=False,
            description="Target variable for prediction",
        )

        # Add other expected columns
        additional_columns = {
            "arrival_date": Column(
                pa.Int64,
                Check.in_range(1, 31, error="Arrival date must be between 1-31"),
                nullable=False,
                description="Day of arrival (1-31)",
            ),
            "arrival_month": Column(
                pa.Int64,
                Check.in_range(1, 12, error="Arrival month must be between 1-12"),
                nullable=False,
                description="Month of arrival (1-12)",
            ),
            "arrival_year": Column(
                pa.Int64,
                Check.in_range(2000, 2030, error="Arrival year must be reasonable"),
                nullable=False,
                description="Year of arrival",
            ),
            "repeated_guest": Column(
                pa.Int64,
                Check.isin([0, 1], error="Repeated guest must be 0 or 1"),
                nullable=False,
                description="Binary indicator for repeat guest",
            ),
            "required_car_parking_space": Column(
                pa.Int64,
                Check.isin([0, 1], error="Car parking space must be 0 or 1"),
                nullable=False,
                description="Binary indicator for parking requirement",
            ),
            "avg_price_per_room": Column(
                pa.Float64,
                [
                    Check.ge(0, error="Price must be non-negative"),
                    Check.le(10000, error="Price seems unrealistic (>$10,000)"),
                ],
                nullable=False,
                description="Average price per room in currency units",
            ),
            "no_of_special_requests": Column(
                pa.Int64,
                [
                    Check.ge(0, error="Special requests must be non-negative"),
                    Check.le(20, error="Too many special requests (>20)"),
                ],
                nullable=False,
                description="Number of special requests made",
            ),
        }

        # Merge without duplicates
        for col, column_schema in additional_columns.items():
            if col not in schema_columns:
                schema_columns[col] = column_schema

        return DataFrameSchema(
            columns=schema_columns,
            checks=[
                # Removed the duplicate check since we expect duplicates might exist
                # after data type conversion but before final deduplication
                Check(lambda df: len(df) > 0, error="DataFrame is empty"),
                Check(
                    lambda df: df.isnull().sum().sum() == 0,
                    error="DataFrame contains null values",
                ),
            ],
            strict=False,  # Allow additional columns
            description="Schema for validated hotel reservation data",
        )

    def _get_numerical_column_schema(self, col: str) -> Column:
        """Get domain-specific schema for numerical columns."""
        if "price" in col.lower():
            return Column(
                pa.Float64,
                [
                    Check.ge(0, error=f"{col} must be non-negative"),
                    Check.le(50000, error=f"{col} seems unrealistic"),
                ],
                nullable=False,
                description=f"Price-related column: {col}",
            )
        elif "nights" in col.lower():
            return Column(
                pa.Int64,
                [
                    Check.ge(0, error=f"{col} must be non-negative"),
                    Check.le(365, error=f"{col} seems unrealistic"),
                ],
                nullable=False,
                description=f"Night count column: {col}",
            )
        elif "time" in col.lower() and "lead" in col.lower():
            return Column(
                pa.Int64,
                [
                    Check.ge(0, error=f"{col} must be non-negative"),
                    Check.le(1000, error=f"{col} seems unrealistic (>1000 days)"),
                ],
                nullable=False,
                description=f"Lead time column: {col}",
            )
        else:
            return Column(
                pa.Int64,
                Check.ge(0, error=f"{col} must be non-negative"),
                nullable=False,
                description=f"Numerical column: {col}",
            )

    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Validate DataFrame against the defined schema."""
        errors: List[str] = []
        warnings: List[str] = []

        try:
            self.schema.validate(df, lazy=True)
            logger.info("✅ Schema validation passed")
            return ValidationResult(True, errors, warnings)
        except Exception as schema_error:
            error_msg = f"Schema validation failed: {str(schema_error)}"
            errors.append(error_msg)
            return ValidationResult(False, errors, warnings)

    def __check_missing_values(
        self,
        df: pd.DataFrame,
        quality_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check for missing values in the DataFrame."""
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            quality_report["issues"].append(
                f"Missing values in columns: {missing_cols}"
            )
        return quality_report

    def __check_infinite_values(
        self,
        df: pd.DataFrame,
        quality_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """check for infinite values in numeric columns."""
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        inf_cols = []
        for col in numeric_cols:
            if df[col].isin([float("inf"), float("-inf")]).any():
                inf_cols.append(col)
        if inf_cols:
            quality_report["issues"].append(f"Infinite values in columns: {inf_cols}")
        return quality_report

    def __detecting_outliers(
        self,
        df: pd.DataFrame,
        quality_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Detect outliers using IQR method."""
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                quality_report["outliers"][col] = len(outliers)
        return quality_report

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality checks."""
        # Unwrap ZenML artifact if necessary
        actual_df = _unwrap_zenml_artifact(df)

        quality_report = {
            "total_rows": len(actual_df),
            "total_columns": len(actual_df.columns),
            "missing_values": actual_df.isnull().sum().to_dict(),
            "duplicate_rows": actual_df.duplicated().sum(),
            "data_types": actual_df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": round(
                actual_df.memory_usage(deep=True).sum() / 1024 / 1024, 2
            ),
            "issues": [],
            "outliers": {},
        }

        # Check for missing values
        quality_report = self.__check_missing_values(actual_df, quality_report)
        # Check for infinite values in numeric columns
        quality_report = self.__check_infinite_values(actual_df, quality_report)
        # Detect outliers using IQR method
        quality_report = self.__detecting_outliers(actual_df, quality_report)

        return quality_report

    def __check_duplicates(self, df: pd.DataFrame) -> None:
        """Check for duplicate rows in the DataFrame."""
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            df = df.drop_duplicates()
            logger.warning(
                f"Found {duplicate_count} duplicate rows in the DataFrame. This suggests the data ingestion pipeline may need to be re-run."
            )
        else:
            logger.info("No duplicate rows found in the DataFrame.")

    def __check_null_values(
        self,
        df: pd.DataFrame,
        errors: List[str],
    ) -> None:
        """Check for null values in the DataFrame."""
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            null_cols = df.columns[df.isnull().any()].tolist()
            errors.append(f"Found null values in columns: {null_cols}")
            # logger.error(f"Found null values in columns: {null_cols}. This suggests the data ingestion pipeline may need to be re-run.")
        else:
            logger.info("No null values found in the DataFrame.")

    def __check_irrelevant_columns(self, df: pd.DataFrame) -> None:
        """Check for irrelevant columns in the DataFrame."""
        irrelevant_patterns = ["booking_id", "Booking_ID", "id"]
        found_irrelevant = [
            col
            for col in df.columns
            if any(pattern in col for pattern in irrelevant_patterns)
        ]
        if found_irrelevant:
            logger.warning(f"Found potentially irrelevant columns: {found_irrelevant}")
        else:
            logger.info("No irrelevant columns found in the DataFrame.")

    def __check_categorical_columns_conversion(
        self,
        df: pd.DataFrame,
        warnings: List[str],
    ) -> None:
        """Check that categorical columns are properly converted to 'category' dtype."""
        for col in self.fe_constants.categorical_columns:
            if col in df.columns and df[col].dtype == "object":
                warnings.append(
                    f"Column {col} is still 'object' dtype - data type conversion may not have been applied"
                )

    def validate_post_ingestion_requirements(
        self, df: pd.DataFrame
    ) -> ValidationResult:
        """Validate that all data ingestion steps were properly applied."""
        errors: List[str] = []
        warnings: List[str] = []

        # Check duplicates - this is critical for data quality
        self.__check_duplicates(df)

        # Check null values were handled
        self.__check_null_values(df, errors)

        # Check irrelevant columns were removed
        self.__check_irrelevant_columns(df)

        # Check that categorical columns are properly converted
        self.__check_categorical_columns_conversion(df, warnings)

        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_business_logic(self, df: pd.DataFrame) -> ValidationResult:
        """Validate business-specific logic and constraints."""
        errors: List[str] = []
        warnings: List[str] = []

        # Check total nights calculation
        if "no_of_weekend_nights" in df.columns and "no_of_week_nights" in df.columns:
            zero_nights = (
                df["no_of_weekend_nights"] + df["no_of_week_nights"] == 0
            ).sum()
            if zero_nights > 0:
                warnings.append(f"Found {zero_nights} bookings with 0 total nights")

        # Check price reasonableness
        if "avg_price_per_room" in df.columns:
            very_low_price = (df["avg_price_per_room"] < 10).sum()
            very_high_price = (df["avg_price_per_room"] > 5000).sum()

            if very_low_price > 0:
                warnings.append(
                    f"Found {very_low_price} bookings with very low prices (<$10)"
                )
            if very_high_price > 0:
                warnings.append(
                    f"Found {very_high_price} bookings with very high prices (>$5000)"
                )

        # Check lead time reasonableness
        if "lead_time" in df.columns:
            extreme_lead_time = (df["lead_time"] > 365).sum()
            if extreme_lead_time > 0:
                warnings.append(
                    f"Found {extreme_lead_time} bookings with lead time > 1 year"
                )

        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_feature_engineering_readiness(
        self, df: pd.DataFrame
    ) -> ValidationResult:
        """Validate that data is ready for feature engineering."""
        errors = []
        warnings = []

        # Check all expected columns are present
        expected_columns = (
            self.fe_constants.categorical_columns
            + self.fe_constants.numerical_columns
            + [self.fe_constants.target_column]
        )

        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing expected columns: {missing_columns}")

        # Check data types alignment - after data ingestion, categorical columns should be 'category' dtype
        for col in self.fe_constants.categorical_columns:
            if col in df.columns and df[col].dtype != "category":
                warnings.append(
                    f"Column {col} has unexpected dtype: {df[col].dtype}. Expected 'category' after data ingestion."
                )

        for col in self.fe_constants.numerical_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column {col} is not numeric: {df[col].dtype}")

        return ValidationResult(len(errors) == 0, errors, warnings)

    def validate_comprehensive(
        self,
        df: pd.DataFrame,
        strict: bool = True,
        auto_clean: bool = False,
    ) -> pd.DataFrame:
        """
        Run comprehensive validation pipeline.

        Args:
            df: DataFrame to validate
            strict: If True, raises exception on validation failure
            auto_clean: If True, attempts to automatically fix common issues

        Returns:
            pd.DataFrame: Validated (and potentially cleaned) DataFrame

        Raises:
            DataValidationError: If validation fails and strict=True
        """
        logger.info("🔍 Starting comprehensive data validation...")

        # Auto-clean if requested and common issues are detected
        if auto_clean:
            duplicate_count = df.duplicated().sum()
            null_count = df.isnull().sum().sum()

            if duplicate_count > 0 or null_count > 0:
                logger.warning(
                    f"⚠️ Auto-clean enabled. Found {duplicate_count} duplicates and {null_count} null values"
                )
                df = self.auto_clean_data(df)

        all_errors = []
        all_warnings = []

        # 1. Schema validation
        schema_result = self.validate_schema(df)
        all_errors.extend(schema_result.errors)
        all_warnings.extend(schema_result.warnings)

        # 2. Data quality validation
        quality_report = self.validate_data_quality(df)
        logger.info(
            f"📊 Data summary: {quality_report['total_rows']:,} rows, "
            f"{quality_report['total_columns']} columns, "
            f"{quality_report['memory_usage_mb']} MB"
        )

        # Add quality issues as warnings
        all_warnings.extend(quality_report["issues"])
        if quality_report["outliers"]:
            all_warnings.append(f"Outliers detected: {quality_report['outliers']}")

        # 3. Post-ingestion validation
        ingestion_result = self.validate_post_ingestion_requirements(df)
        all_errors.extend(ingestion_result.errors)
        all_warnings.extend(ingestion_result.warnings)

        # 4. Business logic validation
        business_result = self.validate_business_logic(df)
        all_errors.extend(business_result.errors)
        all_warnings.extend(business_result.warnings)

        # 5. Feature engineering readiness
        fe_result = self.validate_feature_engineering_readiness(df)
        all_errors.extend(fe_result.errors)
        all_warnings.extend(fe_result.warnings)

        # Generate final result
        final_result = ValidationResult(len(all_errors) == 0, all_errors, all_warnings)
        logger.info(final_result.report)

        # Log individual warnings and errors
        for warning in all_warnings:
            logger.warning(f"⚠️ {warning}")

        for error in all_errors:
            logger.error(f"❌ {error}")

        if all_errors and strict:
            raise DataValidationError(
                f"Validation failed with {len(all_errors)} errors"
            )

        if not all_errors:
            logger.info("🎉 Data validation completed successfully!")

        return df

    def auto_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically clean common data issues found during validation.

        Args:
            df: DataFrame with potential issues

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info("🧹 Auto-cleaning data issues...")

        # Remove duplicates
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df_cleaned)
        if removed_duplicates > 0:
            logger.info(f"🗑️ Removed {removed_duplicates} duplicate rows")

        # Handle any remaining null values
        null_cols = df_cleaned.columns[df_cleaned.isnull().any()].tolist()
        if null_cols:
            logger.warning(f"⚠️ Found remaining null values in: {null_cols}")
            for col in null_cols:
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                else:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            logger.info("🔧 Filled remaining null values")

        return df_cleaned


# Convenience functions for backward compatibility
@step
def validate_cleaned_data(
    df: pd.DataFrame,
    strict: bool = True,
    auto_clean: bool = False,
) -> pd.DataFrame:
    """
    Validate cleaned DataFrame - main entry point for validation.

    Args:
        df: DataFrame to validate (could be ZenML artifact)
        strict: If True, raises exception on validation failure
        auto_clean: If True, attempts to automatically fix common issues like duplicates

    Returns:
        pd.DataFrame: Validated DataFrame
    """
    # Unwrap ZenML artifact if necessary
    actual_df = _unwrap_zenml_artifact(df)

    validator = HotelDataValidator()
    return validator.validate_comprehensive(actual_df, strict, auto_clean)


@step
def generate_validation_report(df: pd.DataFrame) -> str:
    """Generate a detailed validation report for the DataFrame."""
    # Since this is now a ZenML step, df should be unwrapped automatically
    # But let's be safe and unwrap if needed
    actual_df = _unwrap_zenml_artifact(df)

    validator = HotelDataValidator()
    quality_report = validator.validate_data_quality(actual_df)

    report = f"""
📋 DATA VALIDATION REPORT
========================

📊 Basic Statistics:
- Total Rows: {quality_report["total_rows"]:,}
- Total Columns: {quality_report["total_columns"]}
- Memory Usage: {quality_report["memory_usage_mb"]} MB
- Duplicate Rows: {quality_report["duplicate_rows"]}

🔍 Data Quality Issues:
"""

    if quality_report["issues"]:
        for issue in quality_report["issues"]:
            report += f"    - {issue}\n"
    else:
        report += "    ✅ No issues found\n"

    if quality_report["outliers"]:
        report += f"\n    📈 Outliers: {quality_report['outliers']}\n"

    report += f"\n🎯 Data is ready for feature engineering: {'✅ Yes' if not quality_report['issues'] else '⚠️ With warnings'}"

    return report


# For ZenML pipeline integration
def validate_data_with_pandera(df: pd.DataFrame) -> pd.DataFrame:
    """ZenML-compatible validation function."""
    return validate_cleaned_data(df)

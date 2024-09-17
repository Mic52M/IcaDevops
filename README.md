# IcaDevops
 This script applies Independent Component Analysis (ICA) to a dataset to extract independent components and evaluate their quality using kurtosis. The kurtosis values help determine whether the dataset is well-suited for further analysis, based on how "non-Gaussian" the independent components are.


## Overview

This `ICADatasetProbe` is designed for MLOps scenarios where automated dataset analysis is crucial for evaluating data quality before proceeding with machine learning workflows. It integrates with GitLab and GitHub CI pipelines to retrieve datasets and apply Independent Component Analysis (ICA) as a preprocessing step. The core functionality of this probe focuses on extracting independent components from the dataset and using kurtosis as a metric to assess the quality of these components, which helps in determining the "non-Gaussian" nature of the data.

## Core Functionality

### Independent Component Analysis (ICA)
ICA is applied to the dataset to identify independent components, which are transformed versions of the original features. These components represent statistically independent signals, which can be crucial for tasks such as blind source separation or for improving the quality of machine learning models.

The steps for ICA in this probe include:
1. **Data Standardization**: Before applying ICA, the dataset is scaled using `StandardScaler` to ensure that all features have the same scale.
2. **ICA Transformation**: The probe uses `FastICA` to extract up to 20 independent components from the dataset, depending on its size.

### Kurtosis Metric
Kurtosis is a statistical measure used to evaluate the "non-Gaussianity" of the independent components extracted through ICA. The key aspects are:
- **High kurtosis** (values > 1 or < -1) indicates that the components are non-Gaussian and likely independent.
- **Low kurtosis** (values close to 0) suggests that the components may still be Gaussian and not sufficiently independent.

The probe computes kurtosis values for each independent component and evaluates the overall quality of the dataset based on these values. If all components exhibit high kurtosis, the dataset is deemed to be good for analysis; otherwise, it may not be optimal for further processing.

## Git Integration

The probe connects to CI systems (GitLab or GitHub) to retrieve datasets automatically:
- **GitLab**: Downloads the dataset artifact based on the provided branch name and job name.
- **GitHub**: Fetches the dataset artifact based on the specified artifact name from a repository.

These datasets are processed and analyzed using the ICA and kurtosis methods described above.

## Key Methods

- **`parse_input()`**: Reads and parses configuration details such as the repository type (GitLab or GitHub), project details, artifact path, and label columns from the input.
- **`load_and_prepare_dataset()`**: Retrieves the dataset artifact from the configured Git repository, checks for the presence of label columns, and separates the features (`X`) from the labels (`y`).
- **`apply_ica(X, n_components)`**: Applies the ICA algorithm to the feature matrix `X` and returns the transformed independent components.
- **`evaluate_dataset(X_transformed)`**: Evaluates the transformed dataset by computing the kurtosis of each independent component. It determines whether the dataset is suitable for further analysis based on these kurtosis values.

## Error Handling

The probe includes detailed error handling for various scenarios:
- **GitLab and GitHub Authentication**: If the credentials are incorrect or the authentication fails, appropriate errors are captured and logged.
- **Artifact Retrieval Issues**: Errors related to missing or inaccessible artifacts are handled, and the user is notified.
- **Label Column Errors**: If the expected label column is missing from the dataset, a clear error message is provided.

## Usage in MLOps

This probe is built to be integrated into continuous integration (CI) pipelines for machine learning operations (MLOps). It automates the process of dataset quality evaluation by pulling artifacts from version-controlled repositories, running the ICA transformation, and assessing the kurtosis metric. This ensures that only datasets with high-quality independent components are used for model training or further analysis.


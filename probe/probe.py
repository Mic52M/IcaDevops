#!/usr/bin/env python
import pandas as pd
from mooncloud_driver import abstract_probe, atom, result, entrypoint
from git_ci import gitCI
import gitlab
import github
from github import GithubException
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
import typing

class ICADatasetProbe(abstract_probe.AbstractProbe):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.git_ci = None

    def requires_credential(self) -> any:
        return True

    def parse_input(self):
        config = self.config.input.get("config", {})
        self.host = config.get('target')
        self.repo_type = config.get('repo_type', '').lower()
        self.project = config.get('project')
        self.branch = config.get('branch', 'master')
        self.artifact_path = config.get('artifact_path')
        self.job_name = config.get('job_name') if self.repo_type == "gitlab" else None
        self.artifact_name = config.get('artifact_name') if self.repo_type == "github" else None
        self.label_columns = config.get('label_columns', [])

    def setup_git_ci(self):
        if self.repo_type == "gitlab":
            self.git_ci = gitCI(ci_type=gitCI.CIType.GITLAB, gl_domain=self.host, gl_token=self.config.credential.get('token'), gl_project=self.project)
        elif self.repo_type == "github":
            self.git_ci = gitCI(ci_type=gitCI.CIType.GITHUB, gh_domain=self.host, gh_token=self.config.credential.get('token'), gh_repo=self.project)

    def load_and_prepare_dataset(self):
        self.setup_git_ci()
        artifact_file_path = self.git_ci.getArtifact(branch_name=self.branch, job_name=self.job_name, artifact_path=self.artifact_path, artifact_name=self.artifact_name)

        df = pd.read_csv(artifact_file_path)
        if not self.label_columns:
            self.label_columns = [df.columns[-1]]
        else:
            for label in self.label_columns:
                if label not in df.columns:
                    raise ValueError(f"The column '{label}' does not exists in the dataset.")
        y = df[self.label_columns].values
        if len(self.label_columns) == 1:
            y = y.ravel()
        X = df.drop(self.label_columns, axis=1).values
        return X, y

    def apply_ica(self, X, n_components=None):
        X_scaled = StandardScaler().fit_transform(X)
        ica = FastICA(n_components=n_components)
        X_transformed = ica.fit_transform(X_scaled)
        return X_transformed

    def evaluate_dataset(self, X_transformed):
        kurtosis_vals = kurtosis(X_transformed, fisher=True, axis=0)
        kurtosis_details = {
            "num_independent_components": len(kurtosis_vals),
            "kurtosis_values": kurtosis_vals.tolist()
        }
        if all(k > 1 or k < -1 for k in kurtosis_vals):
            self.result.integer_result = result.INTEGER_RESULT_TRUE
            self.result.pretty_result = "The dataset is considered good."
            kurtosis_details["evaluation"] = "All independent components have a kurtosis value greater than 1 or less than -1."
        else:
            self.result.integer_result = result.INTEGER_RESULT_FALSE
            self.result.pretty_result = "The dataset might not be optimal."
            kurtosis_details["evaluation"] = "Some independent components have a kurtosis value of 1, -1, or close to 0."

        self.result.put_extra_data("kurtosis_details", kurtosis_details)

    def run_analysis(self, inputs: any) -> bool:
        X, y = self.load_and_prepare_dataset()
        X_ica = self.apply_ica(X, n_components=min(20, X.shape[1]))
        self.evaluate_dataset(X_ica)
        return True

    def atoms(self) -> typing.Sequence[atom.AtomPairWithException]:
        return [
            atom.AtomPairWithException(
                forward=self.parse_input,
                forward_captured_exceptions=[]
            ),
            atom.AtomPairWithException(
                forward=self.load_and_prepare_dataset,
                forward_captured_exceptions=[
                    atom.PunctualExceptionInformationForward(
                        exception_class=ValueError,
                        action=atom.OnExceptionActionForward.STOP,
                        result_producer=self.handle_label_column_exception
                    ),
                    atom.PunctualExceptionInformationForward(
                        exception_class=gitlab.GitlabAuthenticationError,
                        action=atom.OnExceptionActionForward.STOP,
                        result_producer=self.handle_gitlab_auth_error
                    ),
                    atom.PunctualExceptionInformationForward(
                        exception_class=gitlab.GitlabGetError,
                        action=atom.OnExceptionActionForward.STOP,
                        result_producer=self.handle_gitlab_get_error
                    ),
                    atom.PunctualExceptionInformationForward(
                        exception_class=github.GithubException,
                        action=atom.OnExceptionActionForward.STOP,
                        result_producer=self.handle_github_error
                    )
                ]
            ),
            atom.AtomPairWithException(
                forward=self.run_analysis,
                forward_captured_exceptions=[]
            ),
        ]

    def handle_gitlab_auth_error(self, exception):
        pretty_result = "GitLab Authentication Error: Unable to authenticate with GitLab."
        error_details = str(exception)
        return result.Result(
            integer_result=result.INTEGER_RESULT_TARGET_CONNECTION_ERROR,
            pretty_result=pretty_result,
            base_extra_data={"Error": error_details}
        )

    def handle_gitlab_get_error(self, exception):
        pretty_result = "GitLab Get Error: Unable to retrieve data from GitLab."
        error_details = str(exception)
        return result.Result(
            integer_result=result.INTEGER_RESULT_TARGET_CONNECTION_ERROR,
            pretty_result=pretty_result,
            base_extra_data={"Error": error_details}
        )

    def handle_github_error(self, exception):
        pretty_result = "GitHub Error: Unable to process GitHub request."
        error_details = str(exception)
        return result.Result(
            integer_result=result.INTEGER_RESULT_TARGET_CONNECTION_ERROR,
            pretty_result=pretty_result,
            base_extra_data={"Error": error_details}
        )

    def handle_label_column_exception(self, e):
        pretty_result = f"Configuration Error: {str(e)}"
        return result.Result(
            integer_result=result.INTEGER_RESULT_INPUT_ERROR,
            pretty_result=pretty_result,
            base_extra_data={"Error": "no label column found"}
        )

if __name__ == '__main__':
    entrypoint.start_execution(ICADatasetProbe)


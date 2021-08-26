"""
The ``mlflow.sas`` module provides an API for logging and loading SAS models. This
module exports SAS models with the following flavor:

aStore format:
<https://documentation.sas.com/doc/en/masag/5.1/n1oajxsgx5pvbrn10bpdufb1axjz.htm#p01q7gm29i0smfn1id3oq32a102t>


"""
import os
import logging
import swat
import yaml
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, INTERNAL_ERROR
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.autologging_utils import _get_new_training_session_class
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

FLAVOR_NAME = "sas"

SERIALIZATION_FORMAT_ASTORE = "astore"
SUPPORTED_SERIALIZATION_FORMATS = [SERIALIZATION_FORMAT_ASTORE]

_logger = logging.getLogger(__name__)
_SASTrainingSession = _get_new_training_session_class()


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """

    import swat
    pip_deps = ["swat=={}".format(swat.__version__)]

    return _mlflow_conda_env(additional_pip_deps=pip_deps, additional_conda_channels=None)


def _start_CAS_session(hostname, username, password, **kwargs):
    """
    Returns a CAS session.

    :param hostname: Name of the host to connect to.
    :param username: Name of the user.
    :param password: User's password.

    """

    return swat.CAS(hostname=hostname, username=username, password=password, **kwargs)


def prepare_CAS_session(hostname, username, password, check_status=False, **kwargs):
    """
    Loads action sets and checks connection status.
    :param check_status: Prints status server; default: False.
    :param hostname: Name of the host to connect to.
    :param username: Name of the user.
    :param password: User's password.
    """

    session = _start_CAS_session(hostname, username, password, **kwargs)
    session.loadactionset('astore')

    connection_status = session.serverstatus()
    if check_status:
        print(connection_status)

    return session


def end_cas_session(session):
    session.terminate()


def _load_model(session, input_path, name_cas, caslib, serialization_format):
    """
    Loads aStore from local directory.
    :param session: CAS session.
    :param input_path: Local path; where aStore file is stored.
    :param name_cas: Name of the aStore file on CAS server.
    :param caslib: Name of the caslib we want to save aStore in.
    :param serialization_format: The format in which the model was serialized. This should be
    ``mlflow.sas.SERIALIZATION_FORMAT_ASTORE``
    """

    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                "Unrecognized serialization format: {serialization_format}. Please specify one"
                " of the following supported formats: {supported_formats}.".format(
                    serialization_format=serialization_format,
                    supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
                )
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    with open(input_path, 'rb') as file:
        blob = file.read()
    file.close()

    store = swat.blob(blob)
    session.astore.upload(rstore=dict(name=name_cas, caslib=caslib, replace=True), store=store)


def _save_model(cas_model_table, model_type, name_cas, caslib, session, output_path, serialization_format):
    """ Saves model CAS table to aStore and downloads file to the local server.

    :param cas_model_table: str name of CAS Table containing the model.
    :param model_type: One of the following model types:
                        -decisiontree
                        -deeplearn
                        -reinforcementlearn
                        -forest
                        -gbtree
                        -...
    :param name_cas: Name of the aStore file on CAS server.
    :param caslib: Name of the caslib we want to save aStore in.
    :param session: CAS session.
    :param output_path: Local path; where we want to save aStore.
    :param serialization_format: The format in which the model was serialized. This should be
    ``mlflow.sas.SERIALIZATION_FORMAT_ASTORE``
    """
    if serialization_format == SERIALIZATION_FORMAT_ASTORE:
        cas_model_table = session.CASTable(name=cas_model_table, caslib=caslib, replace=False)
        astore_file = session.CASTable(name=name_cas, caslib=caslib, replace=True)

        if model_type == "dtree":
            session.loadactionset("decisiontree")
            session.decisiontree.dtreeExportModel(modeltable=cas_model_table, casout=astore_file)
        if model_type == "deeplearn":
            session.loadactionset("deeplearn")
            session.deepLearn.dlExportModel(modeltable=cas_model_table, casout=astore_file)
        if model_type == "reinforcementlearn":
            session.loadactionset("reinforcementLearn")
            session.reinforcementLearn.rlExportModel(modeltable=cas_model_table, casout=astore_file)

        store = session.aStore.download(rstore=dict(name=name_cas, caslib=caslib))

        with open(output_path, "wb") as file:
            file.write(store['blob'])
        file.close()

        # TODO This function only applies to dtree models, otherwise output table is defined when
        #  training the model. How to pass 'savestate' argument to the training function?

    else:
        raise MlflowException(
            message="Unrecognized serialization format: {serialization_format}".format(
                serialization_format=serialization_format),
            error_code=INTERNAL_ERROR)


def save_model(
        cas_model_table,
        model_type,
        name_cas,
        path,
        session,
        caslib,
        conda_env=None,
        mlflow_model=None,
        serialization_format=SERIALIZATION_FORMAT_ASTORE,
        signature: ModelSignature = None,
        input_example: ModelInputExample = None,
):
    """
    Save a SAS model to a path on the local file system. Produces an MLflow Model
    containing the following flavor:

        - :py:mod:`mlflow.sas`

    :param model_type: ...
    :param cas_model_table: sas model to be saved (stored in CAS table).
    :param name_cas: Name of the aStore file on CAS server.
    :param session: CAS session.
    :param caslib: Name of the caslib we want to save aStore in.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If `None`, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'scikit-learn=0.19.2'
                            ]
                        }

    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the formats listed in
                                 ``mlflow.sas.SUPPORTED_SERIALIZATION_FORMATS``.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.

    """

    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                "Unrecognized serialization format: {serialization_format}. Please specify one"
                " of the following supported formats: {supported_formats}.".format(
                    serialization_format=serialization_format,
                    supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
                )
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    if os.path.exists(path):
        raise MlflowException(
            message="Path '{}' already exists".format(path), error_code=RESOURCE_ALREADY_EXISTS
        )
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    model_data_subpath = "model.sasast"

    _save_model(
        cas_model_table=cas_model_table,
        model_type=model_type,
        name_cas=name_cas,
        caslib=caslib,
        session=session,
        output_path=os.path.join(path, model_data_subpath),
        serialization_format=serialization_format
    )

    conda_env_subpath = "conda.yaml"

    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        swat_version=swat.__version__,
        serialization_format=serialization_format,
    )

    # save mlflow_model to path/MLmodel
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def load_model(model_uri, session, name_cas, caslib):
    """ Load a SAS model from a local file.

    :param model_uri: The location, in URI format, of the MLflow model.
    :param session: CAS session.
    :param name_cas: Name of the CAS table we want to store the model in.
    :param caslib: Name of the caslib with desired CAS table.

    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    sas_model_artifacts_path = os.path.join(local_model_path, flavor_conf["pickled_model"])
    serialization_format = flavor_conf.get("serialization_format", SERIALIZATION_FORMAT_ASTORE)
    return _load_model(session=session,
                       input_path=sas_model_artifacts_path,
                       name_cas=name_cas,
                       caslib=caslib,
                       serialization_format=serialization_format)


def log_model(
        cas_model_table,
        artifact_path,
        conda_env=None,
        serialization_format=SERIALIZATION_FORMAT_ASTORE,
        registered_model_name=None,
        signature: ModelSignature = None,
        input_example: ModelInputExample = None,
        await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        **kwargs
):
    """
    Log a SAS model as an MLflow artifact for the current run. Produces an MLflow Model
    containing the following flavor:

        - :py:mod:`mlflow.sas`

    :param cas_model_table: scikit-learn model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If `None`, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'scikit-learn=0.19.2'
                            ]
                        }

    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the formats listed in
                                 ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The Cloudpickle
                                 format, ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
                                 provides better cross-system compatibility by identifying and
                                 packaging code dependencies with the serialized model.
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.


    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.sas,
        sas_model=cas_model_table,
        conda_env=conda_env,
        serialization_format=serialization_format,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        **kwargs
    )


def autolog():
    raise NotImplemented

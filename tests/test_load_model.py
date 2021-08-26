import os
from unittest import TestCase

from mlflow_sas import *


class LoadModelTest(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.cas_url = os.getenv('CAS_URL', None)
        cls.cas_user = os.getenv('CAS_USER', None)
        cls.cas_password = os.getenv('CAS_PASSWORD', None)
        if not cls.cas_url:
            raise AssertionError(
                'No connection to CAS provided. Please define '
                'CAS_URL environment variable.'
            )
        if not cls.cas_user:
            raise AssertionError(
                'No connection to CAS provided. Please define '
                'CAS_USER environment variable.'
            )
        if not cls.cas_password:
            raise AssertionError(
                'No password for CAS connection provided. Please define '
                'CAS_PASSWORD environment variable.'
            )

    def setUp(self) -> None:
        self.session = prepare_CAS_session(
            hostname=os.getenv('CAS_URL'),
            username=os.getenv('CAS_USER'),
            password=os.getenv('CAS_PASSWORD')
        )

        # model_uri="/Users/splakg/PycharmProjects/MLflow_sas/test"

    def test_load_mdodel(self):
        load_model("MLflow_sas/test", self.session, 'loaded_model', 'casuser')
        # TODO: dodać asserty
        # self.assertEqual()

    def tearDown(self) -> None:
        self.session.close()

import os
from unittest import TestCase

from mlflow_sas import *

class SaveModelTest(TestCase):

    @staticmethod
    def _create_sas_model(session):
        target = 'bad'
        class_inputs = ['reason', 'job']
        class_vars = [target] + class_inputs
        interval_inputs = [
            'clage', 'clno', 'debtinc', 'loan', 'mortdue', 'value',
            'yoj', 'ninq', 'derog', 'delinq'
        ]
        all_inputs = interval_inputs + class_inputs
        session.loadactionset('decisiontree')
        session.upload_file(
            'resources/hmeq.csv', casout='HMEQ'
        )
        session.decisionTree.gbTreeTrain(
            table=dict(name='HMEQ'),
            inputs=all_inputs,
            nominals=class_vars,
            target=target,
            ntree=10,
            nbins=20,
            maxlevel=6,
            casout=dict(name='gb_model', replace=True)
        )

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

    def test_save_model(self):
        self._create_sas_model(self.session)
        save_model(
            cas_model_table='gb_model',
            model_type='dtree',
            name_cas='astore',
            path='.',
            session=self.session,
            caslib='casuser'
        )

    def tearDown(self) -> None:
        self.session.close()

import shutil

import numpy as np
import pandas as pd

from .base_test_class import DartsBaseTestClass
from ..timeseries import TimeSeries
from ..utils import timeseries_generation as tg
from ..metrics import mape
from ..models import NaiveSeasonal, ExponentialSmoothing, ARIMA, Theta, FourTheta, FFT
from ..utils.utils import SeasonalityMode, TrendMode, ModelMode
from ..logging import get_logger

logger = get_logger(__name__)

# (forecasting models, maximum error) tuples
models = [
    (ExponentialSmoothing(), 5.6),
    (ARIMA(0, 1, 1, trend='t'), 17.1),
    (ARIMA(1, 1, 1, trend='t'), 14.2),
    (Theta(), 11.3),
    (Theta(1), 20.2),
    (Theta(-1), 9.8),
    (FourTheta(1), 20.2),
    (FourTheta(-1), 9.8),
    (FourTheta(trend_mode=TrendMode.EXPONENTIAL), 5.5),
    (FourTheta(model_mode=ModelMode.MULTIPLICATIVE), 11.4),
    (FourTheta(season_mode=SeasonalityMode.ADDITIVE), 14.2),
    (FFT(trend='poly'), 11.4),
    (NaiveSeasonal(), 32.4),
]
# forecasting models with exogenous variables support
extended_models = [ARIMA()]

try:
    from ..models import Prophet
    models.append((Prophet(), 13.5))
except ImportError:
    logger.warning('Prophet not installed - will be skipping Prophet tests')

try:
    from ..models import AutoARIMA
    models.append((AutoARIMA(), 13.7))
    extended_models.append(AutoARIMA())
except ImportError:
    logger.warning('pmdarima not installed - will be skipping AutoARIMA tests')

try:
    from ..models import TCNModel
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not installed - will be skipping Torch models tests')
    TORCH_AVAILABLE = False


class LocalForecastingModelsTestCase(DartsBaseTestClass):

    # forecasting horizon used in runnability tests
    forecasting_horizon = 5

    # dummy timeseries for runnability tests
    np.random.seed(1)
    ts_gaussian = tg.gaussian_timeseries(length=100, mean=50)

    # real timeseries for functionality tests
    df = pd.read_csv('examples/AirPassengers.csv', delimiter=",")
    ts_passengers = TimeSeries.from_dataframe(df, 'Month', ['#Passengers'])
    ts_pass_train, ts_pass_val = ts_passengers.split_after(pd.Timestamp('19570101'))

    def test_models_runnability(self):
        for model, _ in models:
            model.fit(self.ts_gaussian)
            prediction = model.predict(self.forecasting_horizon)
            self.assertTrue(len(prediction) == self.forecasting_horizon)

    def test_models_performance(self):
        # for every model, check whether its errors do not exceed the given bounds
        for model, max_mape in models:
            model.fit(self.ts_pass_train)
            prediction = model.predict(len(self.ts_pass_val))
            current_mape = mape(prediction, self.ts_pass_val)
            self.assertTrue(current_mape < max_mape, "{} model exceeded the maximum MAPE of {}."
                            "with a MAPE of {}".format(str(model), max_mape, current_mape))

    def test_multivariate_input(self):
        es_model = ExponentialSmoothing()
        ts_passengers_enhanced = self.ts_passengers.add_datetime_attribute('month')
        with self.assertRaises(ValueError):
            es_model.fit(ts_passengers_enhanced)
        es_model.fit(ts_passengers_enhanced["#Passengers"])
        with self.assertRaises(KeyError):
            es_model.fit(ts_passengers_enhanced["2"])

    def test_exogenous_variables_support(self):
        for model in extended_models:

            # Test models runnability
            model.fit(self.ts_gaussian, exog=self.ts_gaussian)
            prediction = model.predict(
                self.forecasting_horizon,
                exog=tg.gaussian_timeseries(length=self.forecasting_horizon))
            self.assertTrue(len(prediction) == self.forecasting_horizon)

            # Test mismatch in length between exogenous variables and forecasting horizon
            with self.assertRaises(ValueError):
                model.predict(
                    self.forecasting_horizon,
                    exog=tg.gaussian_timeseries(length=self.forecasting_horizon - 1))

            # Test mismatch in time-index/length between series and exogenous variables
            with self.assertRaises(ValueError):
                model.fit(self.ts_gaussian, exog=self.ts_gaussian[:-1])
            with self.assertRaises(ValueError):
                model.fit(self.ts_gaussian[1:], exog=self.ts_gaussian[:-1])

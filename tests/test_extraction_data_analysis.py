"""Tests para src/ML/ExtractionDataAnalysis."""

import sys
from pathlib import Path

import pandas as pd
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "ML" / "ExtractionDataAnalysis"
sys.path.insert(0, str(MODULE_DIR))

from compare_population_of_countries import buildComparisonFrame  # noqa: E402
from outliers_detection import findOutliers  # noqa: E402
from read_countries import COUNTRIES_URL, loadCountries, summarizeCountries  # noqa: E402
from read_countries_correlation import computeCorrelation  # noqa: E402


def testFindOutliersDetectsExtremeValue():
  data = pd.DataFrame({"area": [10.0] * 20 + [1000.0]})
  anomalies = findOutliers(data)
  assert 20 in anomalies
  assert len(anomalies) >= 1


def testFindOutliersEmptyWhenUniform():
  data = pd.DataFrame({"area": [10.0, 10.0, 10.0, 10.0]})
  assert findOutliers(data) == []


def testBuildComparisonFrameWithInMemoryCsv():
  dataFrame = pd.DataFrame(
    {
      "country": ["Spain", "Spain", "France", "France"],
      "year": [2000, 2001, 2000, 2001],
      "population": [40, 41, 60, 61],
    }
  )
  result = buildComparisonFrame(dataFrame, "Spain", "France")
  assert list(result.columns) == ["Spain", "France"]
  assert list(result.index) == [2000, 2001]
  assert result.loc[2000, "Spain"] == 40
  assert result.loc[2001, "France"] == 61


def testBuildComparisonFrameInvalidCountry():
  dataFrame = pd.DataFrame(
    {"country": ["Spain"], "year": [2000], "population": [40]}
  )
  with pytest.raises(ValueError, match="Atlantis"):
    buildComparisonFrame(dataFrame, "Spain", "Atlantis")


def _canReachCountriesUrl() -> bool:
  try:
    import urllib.request

    with urllib.request.urlopen(COUNTRIES_URL, timeout=10) as response:
      return response.status == 200
  except Exception:
    return False


@pytest.mark.skipif(not _canReachCountriesUrl(), reason="Sin acceso a COUNTRIES_URL")
def testLoadCountriesSmoke():
  dataFrame = loadCountries()
  summary = summarizeCountries(dataFrame)
  assert summary["shape"][0] > 0
  assert summary["shape"][1] > 0


@pytest.mark.skipif(not _canReachCountriesUrl(), reason="Sin acceso a COUNTRIES_URL")
def testComputeCorrelationSmoke():
  dataFrame = loadCountries()
  corr = computeCorrelation(dataFrame)
  assert not corr.empty
  assert corr.shape[0] == corr.shape[1]

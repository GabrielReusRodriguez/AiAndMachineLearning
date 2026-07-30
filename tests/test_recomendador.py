"""Tests para src/ML/Recomendador."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "ML" / "Recomendador"
sys.path.insert(0, str(MODULE_DIR))

from recomendador import (  # noqa: E402
  CSV_RATINGS_FILE,
  CSV_REPOS_FILE,
  CSV_USERS_FILE,
  DEFAULT_N_NEIGHBORS,
  N_FIRST_ROWS,
  buildUserItemMatrix,
  countUniqueUsersAndRepos,
  evaluateMse,
  fitUserNeighbors,
  findSimilarUsers,
  loadRatings,
  loadRepos,
  loadUsers,
  predictRating,
  recommendRepos,
)


def testLoadCsvFilesHaveExpectedColumns():
  dfRatings = loadRatings(CSV_RATINGS_FILE)
  dfRepos = loadRepos(CSV_REPOS_FILE)
  dfUsers = loadUsers(CSV_USERS_FILE)
  assert set(dfRatings.columns) == {"userId", "repoId", "rating"}
  assert {"repoId", "title", "categories", "stars"}.issubset(dfRepos.columns)
  assert set(dfUsers.columns) == {"userId", "username", "name"}
  assert len(dfRatings) > 0


def testCountUniqueUsersAndRepos():
  dfRatings = loadRatings(CSV_RATINGS_FILE)
  nUsers, nRepos = countUniqueUsersAndRepos(dfRatings)
  assert nUsers == 30
  assert nRepos == 167


def testBuildUserItemMatrixShapeAndZeros():
  dfRatings = pd.DataFrame(
    {
      "userId": [1, 1, 2, 2],
      "repoId": [10, 20, 10, 30],
      "rating": [5, 3, 4, 2],
    }
  )
  matrix = buildUserItemMatrix(dfRatings)
  assert matrix.shape == (2, 3)
  assert matrix.loc[1, 10] == 5
  assert matrix.loc[1, 30] == 0
  assert matrix.loc[2, 20] == 0


def testFitUserNeighborsRejectsInvalidK():
  matrix = buildUserItemMatrix(loadRatings(CSV_RATINGS_FILE))
  with pytest.raises(ValueError, match="nNeighbors"):
    fitUserNeighbors(matrix, nNeighbors=0)


def testFindSimilarUsersExcludesSelf():
  dfRatings = loadRatings(CSV_RATINGS_FILE)
  matrix = buildUserItemMatrix(dfRatings)
  model = fitUserNeighbors(matrix, nNeighbors=DEFAULT_N_NEIGHBORS)
  similar = findSimilarUsers(matrix, model, userId=1, nNeighbors=3)
  assert 1 not in similar
  assert len(similar) <= 3
  assert all(userId in matrix.index for userId in similar)


def testPredictRatingReturnsValueForKnownPair():
  dfRatings = pd.DataFrame(
    {
      "userId": [1, 1, 2, 2, 3, 3],
      "repoId": [1, 2, 1, 2, 1, 2],
      "rating": [5, 1, 5, 1, 4, 2],
    }
  )
  matrix = buildUserItemMatrix(dfRatings)
  model = fitUserNeighbors(matrix, nNeighbors=2)
  predicted = predictRating(matrix, model, userId=1, repoId=2, nNeighbors=2)
  assert predicted is not None
  assert 1 <= predicted <= 5


def testRecommendReposOnlyUnrated():
  dfRatings = pd.DataFrame(
    {
      "userId": [1, 1, 2, 2, 3, 3],
      "repoId": [1, 2, 1, 3, 2, 3],
      "rating": [5, 4, 5, 3, 4, 5],
    }
  )
  matrix = buildUserItemMatrix(dfRatings)
  model = fitUserNeighbors(matrix, nNeighbors=2)
  recommendations = recommendRepos(matrix, model, userId=1, topN=5)
  recommendedIds = [repoId for repoId, _ in recommendations]
  assert 1 not in recommendedIds
  assert 2 not in recommendedIds
  assert all(score > 0 for _, score in recommendations)


def testEvaluateMseOnRealData():
  dfRatings = loadRatings(CSV_RATINGS_FILE)
  mse = evaluateMse(dfRatings, testSize=0.2, nNeighbors=5, randomState=42)
  assert mse >= 0
  assert np.isfinite(mse)


def testEvaluateMseRejectsInvalidTestSize():
  dfRatings = loadRatings(CSV_RATINGS_FILE)
  with pytest.raises(ValueError, match="testSize"):
    evaluateMse(dfRatings, testSize=0.0)


def testPipelineConstants():
  assert N_FIRST_ROWS == 10
  assert CSV_RATINGS_FILE.exists()
  assert CSV_REPOS_FILE.exists()
  assert CSV_USERS_FILE.exists()

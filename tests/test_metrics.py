import math

import pytest

from twelve_angry_llms.metrics import (
    ija,
    kendall_tau_b,
    krippendorff_alpha,
    pairwise_winner_agreement,
    spearman_rho,
)


class TestKendallTauB:
    def test_perfect_agreement(self):
        assert kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0

    def test_perfect_disagreement(self):
        assert kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0

    def test_ties_hand_computed(self):
        # x=[1,2,2,3], y=[1,2,3,3]: C=4, D=0, ties_x=1, ties_y=1
        # tau_b = 4 / sqrt(5 * 5) = 0.8
        assert kendall_tau_b([1, 2, 2, 3], [1, 2, 3, 3]) == pytest.approx(0.8)

    def test_constant_vector_undefined(self):
        assert kendall_tau_b([2, 2, 2], [1, 2, 3]) is None

    def test_both_tied_pairs_excluded(self):
        # pairs tied in both vectors contribute to neither denominator
        assert kendall_tau_b([1, 1, 2], [3, 3, 4]) == 1.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            kendall_tau_b([1, 2], [1, 2, 3])


class TestSpearman:
    def test_perfect(self):
        assert spearman_rho([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)

    def test_reversed(self):
        assert spearman_rho([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

    def test_ties_average_ranks(self):
        # x=[1,1,2] -> ranks [1.5,1.5,3]; y=[1,2,2] -> ranks [1,2.5,2.5]
        # Pearson of rank vectors = 0.5
        assert spearman_rho([1, 1, 2], [1, 2, 2]) == pytest.approx(0.5)

    def test_constant_undefined(self):
        assert spearman_rho([5, 5, 5], [1, 2, 3]) is None


class TestPairwiseWinnerAgreement:
    def test_identical(self):
        assert pairwise_winner_agreement([1, 2, 3], [4, 5, 6]) == 1.0

    def test_opposite(self):
        assert pairwise_winner_agreement([1, 2, 3], [3, 2, 1]) == 0.0

    def test_shared_ties_count_as_agreement(self):
        assert pairwise_winner_agreement([1, 1], [2, 2]) == 1.0

    def test_partial(self):
        # pairs: (0,1) both prefer j, (0,2) both prefer j, (1,2) disagree
        assert pairwise_winner_agreement([1, 2, 3], [1, 3, 2]) == pytest.approx(2 / 3)


class TestIJA:
    def test_mean_of_pairs(self):
        vectors = [[1, 2, 3, 4], [1, 2, 3, 4], [4, 3, 2, 1]]
        value, pairs = ija(vectors)
        # pairs: (0,1)=1, (0,2)=-1, (1,2)=-1
        assert value == pytest.approx(-1 / 3)
        assert len(pairs) == 3

    def test_undefined_pairs_skipped(self):
        vectors = [[1, 2, 3], [3, 2, 1], [2, 2, 2]]  # third judge constant
        value, pairs = ija(vectors)
        assert value == pytest.approx(-1.0)
        assert [(i, j) for i, j, _ in pairs] == [(0, 1)]

    def test_no_defined_pairs(self):
        value, pairs = ija([[1, 1], [2, 2]])
        assert value is None
        assert pairs == []

    def test_metric_selection(self):
        vectors = [[1, 2, 3], [1, 3, 2]]
        value, _ = ija(vectors, metric="winner")
        assert value == pytest.approx(2 / 3)

    def test_unknown_metric(self):
        with pytest.raises(ValueError):
            ija([[1, 2]], metric="cosine")


class TestKrippendorffAlpha:
    def test_perfect_agreement(self):
        assert krippendorff_alpha([[1, 2], [1, 2]]) == pytest.approx(1.0)

    def test_systematic_disagreement_hand_computed(self):
        # A=[1,2], B=[2,1]: observed=1.0, expected=2/3, alpha=-0.5
        assert krippendorff_alpha([[1, 2], [2, 1]]) == pytest.approx(-0.5)

    def test_missing_values_ignored(self):
        # middle item has only one value -> not pairable -> dropped; the two
        # remaining items are in perfect agreement
        alpha = krippendorff_alpha([[1, None, 2], [1, 3, 2]])
        assert alpha == pytest.approx(1.0)

    def test_all_identical_undefined(self):
        # zero expected disagreement
        assert krippendorff_alpha([[3, 3], [3, 3]]) is None

    def test_too_sparse(self):
        assert krippendorff_alpha([[1, None], [None, 2]]) is None

    def test_ragged_rejected(self):
        with pytest.raises(ValueError):
            krippendorff_alpha([[1, 2], [1]])

    def test_values_are_finite(self):
        alpha = krippendorff_alpha([[1, 2, 3, 4], [1, 2, 4, 3], [2, 1, 3, 4]])
        assert alpha is not None and math.isfinite(alpha)

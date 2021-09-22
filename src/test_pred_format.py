from nose.tools import assert_equal, assert_in


group_n = 5 # your group number goes here
movie_preds = f'../data/processed/group{group_n}_movie_preds.txt'
game_preds = f'../data/processed/group{group_n}_game_preds.txt'
valid_labels = ['positive', 'negative']


def read_labels(filename):
    with open(filename) as f:
        preds = [line.strip() for line in f.readlines()]
    return preds


def test_movie_preds():
    global movie_preds, valid_labels
    preds = read_labels(movie_preds)
    assert_equal(len(preds), 10000)
    for pred in preds:
        assert_in(pred, valid_labels)


def test_game_preds():
    global game_preds, valid_labels
    preds = read_labels(game_preds)
    assert_equal(len(preds), 21142)
    for pred in preds:
        assert_in(pred, valid_labels)

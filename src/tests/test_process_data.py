"""
This module is created to test functions on
src/tests/process_data.py
"""
import pandas as pd
import pytest
from src.data.process_data import tokenization, get_stemmed_text, process_df


@pytest.mark.parametrize(
    "review, tokenized",
    [
        ('These trousers are so cool!', 'trousers cool'),
        ('I love how this skirt suits me.', 'love skirt suits'),
        (
                'The color could be better, but it is not a bad shirt at all.',
                'color could better bad shirt'
        ),
        ('I completely hate it!', 'completely hate')

    ]
)
def test_tokenization_output(review, tokenized):
    """
    This test checks if, given a review as input,
    tokenization function returns the tokenized
    review as output.
    Args:
        review: text given as input
        tokenized: review after applying tokenization
    """

    data = [[review]]

    dataframe = pd.DataFrame(data, index=[1], columns=['Review Text'])

    dataframe = tokenization(dataframe)

    assert (dataframe['Review Text'] == tokenized).all()


@pytest.mark.parametrize(
    "review, stemmed",
    [
        ('These trousers are so cool!', 'these trouser are so cool!'),
        ('I love how this skirt suits me.', 'i love how this skirt suit me.'),
        (
                'The color could be better, but it is not a bad shirt at all.',
                'the color could be better, but it is not a bad shirt at all.'
        ),
        ('I completely hate it!', 'i complet hate it!')

    ]
)
def test_get_stemmed_text_output(review, stemmed):
    """
    This test checks if, given a review as input,
    get_stemmed_text function returns the stemmed
    review as output.
    Args:
        review: text given as input
        stemmed: review after applying get_stemmed_text
    """

    data = [[review]]

    dataframe = pd.DataFrame(data, index=[1], columns=['Review Text'])

    dataframe = get_stemmed_text(dataframe)

    assert 'Stemmed Review Text' in dataframe.columns
    assert (dataframe['Stemmed Review Text'] == stemmed).all()


@pytest.mark.parametrize(
    "review, tokenized, stemmed",
    [
        ('These trousers are so cool!', 'trousers cool', 'trouser cool'),
        ('I love how this skirt suits me.', 'love skirt suits', 'love skirt suit'),
        (
                'The color could be better, but it is not a bad shirt at all.',
                'color could better bad shirt',
                'color could better bad shirt'
        ),
        ('I completely hate it!', 'completely hate', 'complet hate')

    ]
)
def test_process_df_output(review, tokenized, stemmed):
    """
    This test checks that, if given a text review as input,
    process_df returns a dataframe as output with a column
    with the tokenized text (Review Text) and another one
    with the tokenized and stemmed text (Stemmed Review Text)
    Args:
        review: text given as input
        tokenized: review after applying tokenization function
        stemmed: tokenized after applying get_stemmed_text
    """
    data = [[review]]

    dataframe = pd.DataFrame(data, index=[1], columns=['Review Text'])

    dataframe = process_df(dataframe)

    assert (dataframe['Review Text'] == tokenized).all()
    assert 'Stemmed Review Text' in dataframe.columns
    assert (dataframe['Stemmed Review Text'] == stemmed).all()

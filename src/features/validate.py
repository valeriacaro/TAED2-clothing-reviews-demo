'''
This module is created to validate some data
properties that we must ensure in order to
expect the desired behaviour of these when
giving them as input to the model. The module
is implemented using great_expectations
'''

import great_expectations as gx
import pandas as pd
from src import PROCESSED_TEST_DATA_PATH, PROCESSED_TRAIN_DATA_PATH

if __name__ == '__main__':

    context = gx.get_context()

    context.add_or_update_expectation_suite("reviews_training_suite")

    datasource = context.sources.add_or_update_pandas(name="reviews_dataset")

    train = pd.read_csv(PROCESSED_TRAIN_DATA_PATH/ "train_data.csv")
    test = pd.read_csv(PROCESSED_TEST_DATA_PATH / "test_data.csv")

    processed_data = pd.concat([train, test], axis=0)

    data_asset = datasource.add_dataframe_asset(name="processed", dataframe=processed_data)

    batch_request = data_asset.build_batch_request()
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="reviews_training_suite",
        datasource_name="reviews_dataset",
        data_asset_name="processed",
    )

    validator.expect_column_to_exist("Review Text")
    validator.expect_column_to_exist("Top Product")
    validator.expect_column_to_exist("Stemmed Review Text")

    validator.expect_table_columns_to_match_ordered_list(
        column_list=[
            "Review Text",
            "Top Product",
            "Stemmed Review Text"
        ]
    )

    validator.expect_column_values_to_not_be_null("Top Product")
    validator.expect_column_values_to_be_of_type("Top Product", "int64")
    validator.expect_column_values_to_be_between("Top Product", min_value=0, max_value=1)

    validator.expect_column_values_to_not_be_null("Review Text")
    validator.expect_column_values_to_be_of_type("Review Text", "str")

    validator.expect_column_values_to_not_be_null("Stemmed Review Text")
    validator.expect_column_values_to_be_of_type("Stemmed Review Text", "str")

    validator.save_expectation_suite(discard_failed_expectations=False)

    checkpoint = context.add_or_update_checkpoint(
        name="reviews_checkpoint",
        validator=validator,
    )

    checkpoint_result = checkpoint.run()
    context.view_validation_result(checkpoint_result)

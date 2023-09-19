---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/datasets-cards
{{ card_data }}
---

# Dataset Card for Women's E-Commerce Clothing Reviews

## Dataset Description

- **Homepage:** [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
- **Repository:** [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

  
- **Paper:** {{ paper_url | default("", true)}}
- **Leaderboard:** {{ leaderboard_url | default("", true)}}
- **Point of Contact:** {{ point_of_contact | default("", true)}}

### Dataset Summary

This is a Women’s Clothing E-Commerce dataset revolving around the reviews written by customers. Its nine supportive features offer a great environment to parse out the text through its multiple dimensions. Because this is real commercial data, it has been anonymized, and references to the company in the review text and body have been replaced with “retailer”.

### Supported Tasks and Leaderboards

The Women’s Clothing E-Commerce dataset is designed to support:
- **Sentiment Analysis:** Researchers and data analysts can use this dataset to perform sentiment analysis to understand customer sentiment towards clothing and if they recommend it.
- **Consumer Insights:** Businesses can gain valuable insights into customer preferences and areas for improvement based on the reviews and ratings.
- **Natural Language Processing (NLP):** NLP practitioners can utilize textual reviews for various NLP tasks, such as text classification, summarization, and topic modelling.

### Languages

The Women’s Clothing E-Commerce dataset is primarily focused on the English language. All data instances within this dataset are in English.


## Dataset Structure

### Data Instances

Here are a few sample data instances from the Women’s Clothing E-Commerce dataset to illustrate its content and structure:

| ID              | Clothing ID             | Age                    | Title | Review Text | Rating | Recommended IND | Positive Feedback Count| Division Name | Department Name | Class Name |
| :---------------- | :------:              | :----:                  | :----: | :----: | :----------: | :----------: |:----------: |:----------: |:----------: | ----------: |
|0|767|33|  |Absolutely wonderful - silky and sexy and comfortable|4|1|0|Initmates|Intimate|Intimate|
|1|1080|34|  |Love this dress! it's sooo pretty. i happened to find it in a store, and i'm glad i did bc i never...|5|1|4|General|Dresses|Dresses|
|2|1077|60|Some major design flaws |I had such high hopes for this dress and really wanted it to work for me. i initially ordered the pe...|3|0|0|General|Dresses|Dresses|

### Data Fields

The Women’s Clothing E-Commerce dataset includes the following key data fields, each serving a specific purpose in organizing and describing the data; it includes 23486 rows and 10 feature variables:

1. **Clothing ID:** Integer Categorical variable that refers to the specific piece being reviewed.
2. **Age:** Positive Integer variable that refers to the reviewers age.
3. **Title:** String variable that refers to the title of the review.
4. **Review Text:** String variable for the review body, its content itself.
5. **Rating:** Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
6. **Recommended IND:** Binary variable stating if the customer recommends the product; 1 is recommended, 0 is not recommended.
7. **Positive Feedback Count:** Positive Integer documenting the number of other customers who found this review positive.

8. **Division Name:** Categorical name of the product high level division.
9. **Department Name:** Categorical name of the product department name.
10. **Class Name:** Categorical name of the product class name.


### Data Splits

{{ data_splits_section | default("[More Information Needed]", true)}}

## Dataset Creation

### Curation Rationale

{{ curation_rationale_section | default("[More Information Needed]", true)}}

### Source Data

#### Initial Data Collection and Normalization

{{ data_collection_section | default("[More Information Needed]", true)}}

#### Who are the source language producers?

{{ source_language_producers_section | default("[More Information Needed]", true)}}

### Annotations

#### Annotation process

{{ annotation_process_section | default("[More Information Needed]", true)}}

#### Who are the annotators?

{{ who_are_annotators_section | default("[More Information Needed]", true)}}

### Personal and Sensitive Information

{{ personal_and_sensitive_information_section | default("[More Information Needed]", true)}}

## Considerations for Using the Data

### Social Impact of Dataset

{{ social_impact_section | default("[More Information Needed]", true)}}

### Discussion of Biases

{{ discussion_of_biases_section | default("[More Information Needed]", true)}}

### Other Known Limitations

{{ known_limitations_section | default("[More Information Needed]", true)}}

## Additional Information

### Dataset Curators

{{ dataset_curators_section | default("[More Information Needed]", true)}}

### Licensing Information

{{ licensing_information_section | default("[More Information Needed]", true)}}

### Citation Information

{{ citation_information_section | default("[More Information Needed]", true)}}

### Contributions

{{ contributions_section | default("[More Information Needed]", true)}}

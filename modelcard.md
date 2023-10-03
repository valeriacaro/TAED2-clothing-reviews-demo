---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

# Model Card for Women's E-Commerce Clothing Reviews

The model used is a Long Short-Term Memory (LSTM) model, a type of recurrent neural network (RNN) designed to analyze product reviews and predict whether people are likely to recommend the product or not. 

## Model Details

### Model Description

The LSTM (Long Short-Term Memory) model is a sequential neural network architecture uniquely
designed for opinion analysis and product recommendation prediction. It capitalizes on the inherent
structure of textual data, enabling it to capture intricate dependencies and patterns within reviews.
Consequently, the model can make binary recommendations by considering both the sentiment and
content of input reviews.

To enhance its performance, the LSTM model operates bidirectionally, meaning it processes sequences
both forwards and backwards. This enables it to capture context from both directions and better
understand the nuances in reviews. Moreover a Dropout, with a rate of 0.2, is implemented within
the model to prevent overfitting and improve generalization.


- **Developed by:** Valèria Caro Via, Esther Fanyanàs i Ropero, Claudia Len Manero
- **Shared by [optional]:** {{ shared_by | default("[More Information Needed]", true)}}
- **Model type:** LSTM
- **Language(s) (NLP):** English
- **License:** {{ license | default("[More Information Needed]", true)}}
- **Finetuned from model [optional]:** {{ finetuned_from | default("[More Information Needed]", true)}}

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** {{ repo | default("[More Information Needed]", true)}}
- **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}}
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}}

## Uses

The model is designed to analyze customer reviews and comments in order to understand sentiment and determine whether customers are likely to recommend a product positively or not. 

### Direct Use

This is useful for companies to make data-driven decisions, without having to read all the reviews. An idea of the product recommendations will be available and thus be able to make improvements through a global view of all the reviews.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

{{ downstream_use | default("[More Information Needed]", true)}}

### Out-of-Scope Use

The model has been trained from reviews, thus there is a subjective opinion in the text.

## Bias, Risks, and Limitations

### Bias
- The opinion on a product depends on several subjective aspects that are not covered by the model, such as the size of women.

### Risks


### Limitations 
- The model may be less accurate for text data that is significantly different from the e-commerce reviews it was trained on.
- The model may be less accurate with male review text data, as all instances of the model are female.
- The model has been trained in English, so it does not support the input of text in other languages.
- The model does not interpret emojis.

### Recommendations

The model is recommended for analyzing overall women's trends in customer sentiment and identifying areas for improvement based on customer feedback.

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

## Training Details

### Training Data

The processed data underwent a split, allocating 85% of the data for the training dataset. Within this
training dataset, a further division was made, reserving 70% for the primary training subset and 15%
for data validation.

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

{{ training_data | default("[More Information Needed]", true)}}

### Training Procedure 

During the training process, cross-validation for 10 epochs was applied to iteratively
enhance the model’s performance and robustness. 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

{{ preprocessing | default("[More Information Needed]", true)}}


#### Training Hyperparameters

The hyperparameteres have been evaluated through experiments in MlFlow and the best results
obtained have been with the follows:

- **Batch Size:** 512

- **Embedding Size:** 128

- **Hidden Size:** 256

- **Token Size:** 20000


- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

This section describes the evaluation protocols and provides the results.

### Testing Data, Factors & Metrics

#### Testing Data

or our test dataset, we employed a 15% split of the preprocessed data.

<!-- This should link to a Data Card if possible. -->

{{ testing_data | default("[More Information Needed]", true)}}

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

In assessing the model’s performance, we focused on the Accuracy metric. This choice aligns with our objective of ensuring
comprehensive representation of opinions. We aim to accurately classify reviews, categorizing products
as either recommended or not recommended, encompassing both positive and negative sentiments.
Subsequently, we conducted an evaluation of the model’s performance, resulting in the following
performance metrics:

### Results

Subsequently, we conducted an evaluation of the model’s performance, resulting in the following
performance metrics:

- **Training Accuracy:** 90.78%
- **Validation Accuracy:** 77.5%

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

{{ model_examination | default("[More Information Needed]", true)}}

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** {{ hardware | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}

## Technical Specifications [optional]

### Model Architecture and Objective

{{ model_specs | default("[More Information Needed]", true)}}

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

{{ hardware | default("[More Information Needed]", true)}}

#### Software

{{ software | default("[More Information Needed]", true)}}

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}

**APA:**

{{ citation_apa | default("[More Information Needed]", true)}}

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

{{ glossary | default("[More Information Needed]", true)}}

## More Information [optional]

{{ more_information | default("[More Information Needed]", true)}}

## Model Card Authors [optional]

{{ model_card_authors | default("[More Information Needed]", true)}}

## Model Card Contact

{{ model_card_contact | default("[More Information Needed]", true)}}

# Transformer_Decoder
Implementation of Transformer_Only_Decoder

# Requirements
torch == 2.0.1

numpy = 1.23.5

transformers = 4.35.2

# Collab installations
!pip install transformers datasets


# Description
The Transformer Decoder Only Model is a generative model that is similar to the Transformer Encoder Only Model which implements an embedding layer followed by a positional encoder to keep the embeded data from staying in a sequential format but we also implement the causal attention head which in this case allows the transformer model to have an autoregressive nature. The Transformer Decoder Only Model is famously the architecture that is use in most generative Large Language Models such as the GPT family and are responsible in predicting the next word token based on previous word tokens. In this project we are trying to get a story teller AI that will try to continue a story based on references of the harry potter world.

# Dataset
Harry Potter Books

# Tokenizer
Distilbert

# Architecture
Transfomer Decoder Only Model

# optimizer
Adam

# loss function
Cross Entropy Loss

# Text Result:
![story_harry](https://github.com/Santiagor2230/Transformer_Decoder/assets/52907423/e25434ce-addc-4c35-9b5e-7535dc66bfe7)

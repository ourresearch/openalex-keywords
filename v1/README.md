### v1 of OpenAlex Keywords

Our team put together a quick implementation of a pretrained model/system in order to generate keywords for each paper that comes into OpenAlex. While keywords are not generated for every paper, we think it could still be a useful tool in the future. Our keywords will be improved as time goes on (possibly with a future model upgrade) but we expect that papers will start including keywords as metadata more often so this is our way of getting ready for that to be more prevalent. Work titles were the only data used, we did not use abstracts as well. Testing was done to compare results for titles vs titles + abstracts and much better results were obtained with using titles only.

#### Short Explanation

Unlike our other models (institution parsing, concept tagging, etc.) we used a pretrained model (all-MiniLM-L6-v2), an open-source python library (KeyBERT), as well as a small amount of additional code. While there are recent enhancements to KeyBERT using LLMs, we decided to stick with the more traditional approach due to the high latency and cost associated with running all of our works through an LLM. As mentioned before, not all papers will have keywords. We implemented a score threshold to make sure that only keywords with high scores would make it through. We also used a python library called keyphrase-vectorizers to further improve the results of KeyBERT. Lastly, keywords were filter to only have 4 words or less. With all of these constraints and processes, some works do not get assigned a keyword.

#### Notebooks

Both notebooks that were used to create keywords and load into OpenAlex are provided. However, the only notebook that will be useful for most users is "run_keywords.ipynb". This notebook will contain all of the code for taking work titles and generating keywords. The other notebook takes that data and loads it into an OpenAlex database using PySpark.


#### Problems/Errors

While we think most of the keywords generated will be useful/correct, there is always the chance for incorrect keywords to show up. If you see any issues with keywords, feel free to submit a support request at the following link: [Support Request](https://openalex.org/help)


### NOTES
#### KeyBERT

For more information on KeyBERT, check out the following repo: [MaartenGr/KeyBERT](https://github.com/MaartenGr/KeyBERT)

@misc{grootendorst2020keybert,
  author       = {Maarten Grootendorst},
  title        = {KeyBERT: Minimal keyword extraction with BERT.},
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.3.0},
  doi          = {10.5281/zenodo.4461265},
  url          = {https://doi.org/10.5281/zenodo.4461265}
}

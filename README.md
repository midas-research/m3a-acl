# Code and Data for M3A: Multimodal Multi-speaker Mergers &amp; Acquisitions at ACL-IJCNLP 2021

# Multimodal Multi-Speaker Merger & AcquisitionFinancial Modeling: A New Task, Dataset, and Neural Baselines

This repository contains the code and dataset for M3ANet, the model introduced in Multimodal Multi-Speaker Merger & AcquisitionFinancial Modeling: A New Task, Dataset, and Neural Baselines.

Accepted at the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021)

Environment: Python 3.7 and Tensorflow v2

# Dataset

The dataset folder contains the processed form of all the merger and acquisition calls collected in the work. Each folder has the name X_CompanyName1_CompanyName2_Date where X denotes whether the call was a merger or acquisition, CompanyName1 and CompanyName2 are the two concerned companies and Date is the date when the call took place.

Each folder contains two CSVs Text.csv and Audio.csv. The Text.csv file contains the speaker annotation for each utterance labeled as described in the paper followed by the BERT [5] embedding of the utterance. The Audio.csv file contains the 62 GeMAPSv01b [7] audio features for each utterance as obtained from OpenSMILE [6]. For a call with n utterances, Text.csv has a shape (n, 769) while Audio.csv has a shape (n, 62).

# Obtaining Raw Data

To obtain the raw data, you will need a valid Bloomberg login for the Bloomberg Terminal [4]. After logging in, you will need to search for the keywords ‘merger’ and ‘acquisition’ to retrieve all the calls that had either merger or acquisition in their title. From these calls, we further apply the following filters:
Year: 2016 - 2020
Language: English
Domicile: United States of America
Download the audio and the text transcripts of the retrieved calls to recreate the raw dataset.

# Ethical Considerations

Examining a speaker's tone and speech in conference calls is a well-studied task in past literature [1][2]. Our work focuses only on calls for which companies publicly release transcripts and audio recordings. The data used in our study corresponds to M\&A conference calls of companies in the NASDAQ stock exchange. We acknowledge the presence of gender bias in our study, given the imbalance in the gender ratio of speakers of the calls. We also acknowledge the demographic bias [3] in our study as the companies are organizations within the public stock market of the United States of America and may not generalize directly to non-native speakers.

# References

[1] Yu Qin and Yi Yang. 2019. What you say and how you say it matters: Predicting stock volatility using verbal and vocal cues. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 390–401, Florence, Italy. Association for Computational Linguistics.

[2] Anis Chariri. 2009. Ethical culture and financial reporting: Understanding financial reporting practice within javanese perspective*. Issues In Social And Environmental Accounting, 3.

[3] Ramit Sawhney, Shivam Agarwal, Arnav Wadhwa, and Rajiv Ratn Shah. 2020a. Deep attentive learning for stock movement prediction from social media text and company correlations. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8415–8426, Online. Association for Computational Linguistics.

[4] https://bba.bloomberg.net/

[5] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics

[6] https://pypi.org/project/opensmile/

[7] F. Eyben, K. R. Scherer, B. W. Schuller, J. Sundberg, E. Andŕe, C. Busso, L. Y. Devillers, J. Epps, P. Laukka, S. S. Narayanan, and K. P. Truong.2016. The geneva minimalistic acoustic parameter set (gemaps) for voice research and affective computing. IEEE Transactions on Affective Computing, 7(2):190–202.

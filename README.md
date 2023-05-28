# CreditScoringHack2022
AlphaBank hackathon 2022. 26 place private leaderboard

Participants were offered to predict the debtor's default by his loan portfolio. Loan history was given for each bank's debtors. There is information about something around 5000000 debtors for the last 2 years. The competition's data was strongly anonymized. Continuous features were transformed into categorical through histogram splitting. As a result, each feature has no more than 30 values. Data have no explicit information about time, but debt information was sorted by time. The information in the test sample is in time after the information in the training sample.

I was developing lgbm and lstm-based rnn models. In order to encode historical data to lgbm, I calculated the probability of meeting this loan in the portfolio of a bankrupt debtor for each loan. The meta-features were calculated by the out-of-fold stacking method. We can perceive these features as an assessment of how risky a loan is. That features combined with count-aggregated features yield lgbm's quality comparable with rnn models.

Let's see how some count-aggregation features change their values over time.

![alt text](https://github.com/dkhar08/CreditScoringHack2022/blob/main/pictures/f1.png?raw=true)
![alt text](https://github.com/dkhar08/CreditScoringHack2022/blob/main/pictures/f3.png?raw=true)
![alt text](https://github.com/dkhar08/CreditScoringHack2022/blob/main/pictures/f4.png?raw=true)
![alt text](https://github.com/dkhar08/CreditScoringHack2022/blob/main/pictures/f5.png?raw=true)


As we can see dataset endures a strong "dataset shift". There is a certain amount of resample and reweight classical methods to deal with that problem, but usually, they work only for weak models. No surprise, that they didn't work for my lgbm and nn models. Polynomial decaying weights somewhat improved lgbm's quality but not nn's.

Further investigation showed that we could consider the "Dataset shift" problem as a special case of the domain adaptation problem. For nn models, there are a certain amount of advanced methods like ADDA and DANN. But usually, they work well for CNN, and they didn't work for me. For LSTM there are classical domain adaptation methods like LM and Autoencoder pretraining. Autoencoder pretraining showed a much stronger result than LM pretraining. And also its greatly outperformed lgbm with weight decay. As a result my final model was RNN-baseline with pretraining:).

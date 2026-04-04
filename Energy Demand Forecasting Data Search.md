# **Integrating Geopolitical Sentiment Signals into Temporal Fusion Transformers for Crisis-Resilient Energy Demand Forecasting**

## **The Convergence of Geopolitical Volatility and Energy Grid Stability**

The contemporary energy landscape is increasingly defined by its susceptibility to non-linear disruptions that originate far beyond the traditional boundaries of meteorological patterns or historical load cycles. As global supply chains become more interconnected, the stability of regional energy grids in the United States, such as those managed by the Electric Reliability Council of Texas (ERCOT) or the Bonneville Power Administration (BPAT), is inextricably linked to geopolitical sentiment and international policy uncertainty.1 Traditional forecasting models, which predominantly rely on autoregressive features and weather variables, often fail to capture the abrupt structural breaks in demand caused by "Black Swan" events, trade conflicts, or global health crises.3

The integration of high-frequency geopolitical sentiment signals into advanced machine learning architectures offers a pathway toward more resilient forecasting. Specifically, the Temporal Fusion Transformer (TFT) has emerged as a premier framework for this task due to its ability to synthesize multi-horizon time-series data with static metadata that defines the inherent vulnerabilities of specific Balancing Authorities (BAs).5 By establishing static sensitivity coefficients—such as a grid's dependency on natural gas or its reliance on hydroelectric generation—researchers can calibrate the model’s attention mechanism to prioritize relevant sentiment categories during periods of crisis.7

Building such a system requires a nuanced understanding of data sourcing, signal engineering, and the causal mechanisms that link news narratives to physical energy consumption. The primary challenge lies in identifying free, pre-calculated sentiment data that can be regionalized to the BA level, while ensuring that the resulting signals enhance predictive accuracy rather than introducing noise that skews forecasts during stable periods.9

## **Architectural Foundations: The Temporal Fusion Transformer in Energy Systems**

The Temporal Fusion Transformer (TFT) is an attention-based deep learning model specifically designed for multi-horizon time-series forecasting. Unlike traditional Transformer architectures, the TFT is optimized to handle a heterogeneous mix of inputs, which is essential for energy forecasting where the predictors include static attributes of the grid, known future events (e.g., holidays), and observed historical sequences.3

### **Static Covariate Encoders and Variable Selection**

At the heart of the TFT's resilience is the static covariate encoder, which allows the model to differentiate between Balancing Authorities based on their unique structural characteristics. In a project focused on crisis resilience, these static inputs are represented as sensitivity coefficients. For instance, a BA with a high natural gas dependency is more likely to experience demand volatility during international energy trade disputes.7

The TFT utilizes Variable Selection Networks (VSN) to identify the most significant predictors at each time step. The VSN uses Gated Residual Networks (GRN) to provide non-linear processing, allowing the model to "shut off" irrelevant features. The mathematical formulation of the GRN is given by:

![][image1]  
where ![][image2] is the input, ![][image3] is a context vector from the static covariate encoder, and GLU is the Gated Linear Unit.3 This gating mechanism is crucial for sentiment integration; it ensures that geopolitical signals only influence the forecast when they possess genuine predictive power, thereby preventing the model from being skewed by the inherent noise of daily news cycles.9

### **Temporal Self-Attention and Gating Mechanisms**

The TFT employs a specialized multi-head attention mechanism that allows the model to learn long-term dependencies across different time scales. For energy demand, this means the model can correlate current geopolitical tensions with historical demand patterns observed during similar crises.12 The attention weights provide interpretability, allowing grid operators to see which specific days or sentiment spikes are driving a particular forecast.

Furthermore, the model’s use of persistent temporal patterns alongside transient shocks enables it to maintain a high degree of accuracy during both stable regimes and periods of high stochastic structural variation.14 This dual capability is what distinguishes the TFT from simpler recurrent neural networks (RNNs) or long short-term memory (LSTM) models.3

## **Sources for Pre-Calculated Regional Sentiment Data**

One of the most significant barriers to integrating sentiment into forecasting models is the computational cost of processing millions of news articles. For a project with limited compute power, it is imperative to utilize datasets that provide pre-calculated sentiment scores with high regional granularity.

### **The GDELT Project: Global Knowledge Graph (GKG)**

The GDELT Project (Global Database of Events, Language, and Tone) is perhaps the most comprehensive free resource available for regional sentiment analysis. Supported by Google Jigsaw, GDELT monitors broadcast, print, and web news from nearly every country in over 100 languages.16

The GDELT Global Knowledge Graph (GKG) is particularly valuable because it compiles a list of persons, organizations, locations, and themes from every news report. Most importantly, it provides a pre-calculated "Tone" score for each article. The Tone score is a value ranging from \-100 (extremely negative) to \+100 (extremely positive), calculated by comparing the document's text against a massive dictionary of emotional terms.16

| Feature | GDELT Capability for Energy Research |
| :---- | :---- |
| **Regionality** | Geocodes events to the city or landmark level across the USA.16 |
| **Frequency** | Updates every 15 minutes, allowing for near-real-time forecasting inputs.16 |
| **Sentiment** | Provides pre-calculated Tone, Positive Score, and Negative Score.16 |
| **Thematic Filtering** | Allows extraction of sentiment specifically related to "Energy," "Natural Gas," or "Infrastructure".16 |

By using the GKG, researchers can aggregate the average tone of news articles within a specific US state or a coordinate-defined region to create a time-series sentiment signal that corresponds to a Balancing Authority's footprint.18

### **Economic Policy Uncertainty (EPU) Index**

Developed by researchers at Northwestern University and the University of Chicago, the Economic Policy Uncertainty (EPU) Index is a specialized sentiment signal that tracks the frequency of news coverage related to policy-driven economic stress.20 The index is constructed by searching for articles in leading newspapers that contain terms related to "uncertainty," "economics," and specific government policy areas such as "legislation," "regulation," or the "Federal Reserve".20

The EPU provides several layers of data that are critical for energy forecasting:

* **Daily News-Based EPU:** Available at the national level for the USA, providing a high-frequency signal of general economic anxiety.20  
* **US State-Level EPU:** Specialized indices for individual US states, which can be mapped directly to regional BAs like ERCOT or CAISO.20  
* **Categorical EPU Indices:** These allow for the isolation of specific types of uncertainty, such as "Trade Policy Uncertainty" or "Monetary Policy Uncertainty," which have distinct causal effects on industrial energy demand.20

These datasets are available for free through the Federal Reserve Economic Data (FRED) platform and the policyuncertainty.com website, making them highly accessible for researchers with limited resources.22

## **Identifying Resilient and Appropriately Affecting Signals**

Integrating sentiment data into a Temporal Fusion Transformer requires careful selection of the type of signal to avoid skewing predictions. Raw sentiment polarity is often highly volatile and may not accurately reflect the structural shifts in demand.9

### **Polarity vs. Intensity and Uncertainty**

Recent research suggests that the "Intensity" and "Uncertainty" of a news signal are often more predictive than its simple polarity (positive/negative).9 In the context of energy forecasting:

* **Sentiment Intensity:** Refers to the volume of news articles regarding a specific geopolitical event. A surge in the volume of energy-related news—even if the tone is neutral—can indicate an impending shift in industrial consumption or market behavior.9  
* **Forward-Looking Uncertainty:** Signals derived from terms like "risk," "volatility," or "future" tend to have a stronger correlation with short-term demand fluctuations. These dimensions of sentiment capture the psychological "wait-and-see" attitude that firms adopt during crises.4

### **Mitigating Noise through Signal Engineering**

To ensure that sentiment signals appropriately affect the model, they should be processed to remove high-frequency noise. One effective technique is the use of a 28-day or 30-day moving average, which smooths the signal and highlights the underlying trend in geopolitical risk.26

Furthermore, instead of using the absolute sentiment score, models are often more resilient when trained on the "Delta" or change in sentiment. This approach allows the TFT to focus on *shocks* to the system rather than the baseline level of news tone.5 Causal analysis has shown that while physical indicators like satellite imagery are better for long-term trends, textual sentiment has superior predictive power for short-term demand fluctuations, especially during periods of escalating conflict.1

## **Deriving Static Sensitivity Coefficients for US Balancing Authorities**

To utilize sentiment signals effectively, the TFT must be informed by the structural vulnerabilities of the target grid. These are represented as static covariates that define how sensitive a BA is to specific types of geopolitical news.

### **Balancing Authority Data Sources**

The primary source for identifying these coefficients is the U.S. Energy Information Administration (EIA). The EIA-930 Hourly Balancing Authority Operations Report provides real-time and historical data on net generation, demand, and fuel mix for every BA in the Lower 48 states.28

| Balancing Authority | Key Sensitivity Focus | Data Repository |
| :---- | :---- | :---- |
| **ERCOT (Texas)** | Natural Gas, Solar, Wind | ERCOT Fuel Mix Reports / MORA 7 |
| **BPAT (Northwest)** | Hydroelectric, Inter-regional flows | EIA Grid Monitor 7 |
| **PJM (Mid-Atlantic)** | Coal, Gas, Nuclear | EIA-930 Daily Generation Mix 29 |

### **Case Study: ERCOT Natural Gas Dependency**

ERCOT serves as a prime example of a region where geopolitical sentiment regarding fuel markets is highly relevant. The ERCOT Fuel Mix dashboard categorizes generation into Natural Gas (Gas Steam, Simple Cycle, Combined Cycle), Coal, Nuclear, Wind, and Solar.7 Because natural gas often acts as the marginal fuel source in Texas, any geopolitical tension affecting global LNG markets or domestic gas pipelines will have an outsized impact on ERCOT’s operational costs and demand response.7

A sensitivity coefficient for ERCOT might be defined as: ERCO\_gas\_dependency \= 0.85 This value reflects the proportion of thermal capacity or the historical average of gas in the fuel mix during peak periods.7 In the TFT model, this static coefficient tells the variable selection network to give higher weight to "Energy Market" and "Natural Gas" sentiment signals from GDELT when making predictions for the Texas region.

### **Case Study: BPAT Hydroelectric Dependency**

For the Bonneville Power Administration (BPAT), the primary vulnerability is not fossil fuel prices but water availability and environmental policy. BPAT’s reliance on hydro generation makes it sensitive to regulatory news and climate-related policy uncertainty.7

A sensitivity coefficient for BPAT might be: BPAT\_hydro\_dependency \= 0.70 This coefficient informs the model to pay closer attention to "Environmental Policy" and "Land Management" sentiment categories within the GDELT or EPU datasets.7

## **Mapping News Geography to Grid Boundaries**

One of the complex data engineering tasks in this project is the geospatial alignment of news sentiment with BA footprints. While GDELT provides high-resolution geocoding, BAs often span multiple states or are contained within a single state.16

### **State-to-BA Weighting Matrix**

For BAs that align closely with state borders, such as ERCOT (Texas) or CAISO (California), state-level sentiment signals from the EPU index or GDELT filters are sufficient.20 For multi-state BAs, a weighting matrix should be used based on the percentage of the BA’s load served in each state.

| Balancing Authority | Primary States | Recommended Sentiment Mapping |
| :---- | :---- | :---- |
| **MISO** | IL, IN, IA, MI, MN, MO | Weighted Avg of Mid-West State EPUs 20 |
| **PJM** | PA, NJ, MD, VA, OH | Weighted Avg of Mid-Atlantic News Tone 16 |
| **NEVP** | NV | Nevada State-Level EPU Index 23 |

### **GDELT Filtering via FIPS and Coordinates**

GDELT’s GKG allows for sophisticated filtering using FIPS state codes or centroid coordinates. This enables the creation of "virtual sentiment stations" that aggregate news tone within a 100-mile radius of a BA’s central dispatch office or major industrial load centers. This spatial specificity reduces the noise from unrelated national news and ensures the sentiment signal is "region-based" as requested.16

## **Macroeconomic Mechanisms: Why Sentiment Affects Demand**

To justify the use of sentiment in a professional forecasting report, it is necessary to articulate the causal channels through which news narratives manifest as physical changes in energy demand.

### **The Real-Options Channel**

Heightened policy uncertainty, particularly regarding trade or tax legislation, prompts industrial firms to adopt a "wait-and-see" approach.21 This behavior results in the postponement of new production lines and a decrease in the utilization of existing heavy machinery, leading to a measurable decline in industrial baseload demand.4 The EPU "Categorical Index: Monetary Policy" or "Trade Policy" are excellent proxies for this channel.20

### **The Precautionary Saving Channel**

In the residential sector, negative geopolitical news often amplifies household pessimism. This leads to increased precautionary savings and a reduction in discretionary economic activity.25 For energy demand, this may manifest as changes in home occupancy patterns or a reduction in the use of high-energy-consuming household appliances, particularly in cooling or heating seasons where price sensitivity is high.4

### **The Supply-Demand Sentiment Spillover**

Energy markets are particularly prone to "sentiment spillovers," where uncertainty in international gas or oil markets leads to anticipatory price changes and demand-side conservation efforts.11 During the Russia-Ukraine conflict, for example, news-driven volatility in energy futures led to significant shifts in the merit order of generation in US markets, as utilities attempted to minimize exposure to volatile gas prices.24

## **Methodological Integration into the TFT Pipeline**

Implementing this system involves a four-stage data pipeline that prioritizes pre-calculated metrics to minimize local compute requirements.

### **Stage 1: Data Acquisition (Free Sources)**

* **Grid Data:** Download hourly BA generation and demand data from the EIA Open Data API. Specifically, use the EIA-930 series to calculate historical fuel dependency ratios.28  
* **Sentiment Data:** Access the GDELT Project via Google BigQuery (the first 1TB per month is free) to extract the AvgTone and DocumentIntensity features for specific US state codes.  
* **Uncertainty Data:** Pull daily EPU indices from the FRED API for the relevant states and national categories.22

### **Stage 2: Feature Engineering and Normalization**

To ensure signal resilience, the sentiment and uncertainty data should be transformed into z-scores based on a rolling window. This centers the data and ensures that the TFT’s attention mechanism is responding to *anomalies* rather than static levels.5

![][image4]

### **Stage 3: Static Covariate Definition**

Define the sensitivity coefficients for each BA. These are entered into the TFT as static categorical or continuous variables.

* BA\_ID: (Categorical) e.g., ERCO, BPAT, PJM.  
* Gas\_Sens: (Continuous) e.g., 0.85.  
* Hydro\_Sens: (Continuous) e.g., 0.10.  
* Policy\_Sens: (Continuous) Derived from the BA's historical correlation with EPU shocks.

### **Stage 4: TFT Model Training**

During training, the TFT’s Variable Selection Network will learn to interact the static sensitivity coefficients with the time-varying sentiment signals.3 For example, the model will learn that if Gas\_Sens is high, a negative shock in the "Global Energy Trade" sentiment category should lead to a predicted decrease in demand (or an increase in demand volatility).

## **Comparative Evaluation of Sentiment Sources**

For a project focused on efficiency, selecting the right sentiment engine is paramount. The following table compares the pre-calculated sentiment sources identified in this research.

| Source | Pre-Calculated? | Regionality | Energy Specificity | Cost |
| :---- | :---- | :---- | :---- | :---- |
| **GDELT GKG** | Yes (Tone score) | High (Coordinates) | High (Thematic tags) | Free 16 |
| **EPU Index** | Yes (Index value) | Medium (State) | Medium (Policy tags) | Free 20 |
| **Yahoo Finance** | Yes (via tickers) | Low (Company) | High (Energy stocks) | Free 2 |
| **FinBERT (Kaggle)** | Partially (Embeds) | Low (Text) | High (Financial) | Free 34 |

While FinBERT and other transformer-based models (like GPT-4o) provide higher accuracy, they require significant compute power to run locally.9 GDELT and the EPU Index are the superior choices for this project as they provide the sentiment "already calculated," allowing the local compute to be dedicated entirely to the TFT forecasting model.16

## **Ensuring Model Robustness against Prediction Skew**

A major concern with sentiment integration is the potential to "overfit" to news noise, leading to erratic demand predictions. The TFT’s architecture provides several inherent protections against this, but they must be supplemented by strategic data choices.

### **Use of Gated Residual Networks (GRN)**

The GRN allows the TFT to skip processing of certain features if they are not contributing to a reduction in loss. If the news sentiment on a given day is just "noise" (e.g., celebrity news or sports), the gating mechanism will suppress these inputs, ensuring they do not skew the demand forecast.3

### **Multi-Horizon Quantile Forecasting**

Rather than predicting a single point value, the TFT should be configured to produce quantile forecasts (e.g., 10th, 50th, and 90th percentiles). Geopolitical crises often increase the *uncertainty* of demand rather than just shifting the mean. By examining the spread between the 10th and 90th quantiles, grid operators can assess the risk of "Black Swan" events even if the median forecast remains stable.5

### **Incorporating "Implied Sentiment" from VIX**

In addition to textual sentiment, incorporating "implied sentiment" from financial market volatility indices (like the VIX or energy-sector specific volatility) provides a resilient, low-noise signal. These indices act as a market-validated consensus on uncertainty, providing a useful cross-check for the more volatile news-based tone scores.34

## **Crisis-Resilient Forecasting: Future Outlook**

The transition toward a grid powered by decentralized and weather-dependent resources only increases the necessity of sentiment-aware forecasting. As thermal capacity (representing 85% of current generation) becomes more vulnerable to climate-induced efficiency losses and geopolitical fuel shocks, the "psychological layer" of the grid will become as important as the physical layer.32

Future iterations of this research could leverage "Agentic Generative AI pipelines" to summarize news into even more refined semantic embeddings before feeding them into the TFT.15 However, for the current objective of building a crisis-resilient system with limited resources, the combination of GDELT’s regional tone scores, EPU’s state-level uncertainty indices, and the TFT’s static covariate architecture provides a robust and academically sound framework for enhancing the reliability of the US energy grid.

## **Final Summary of Data Integration**

The following mapping summarizes the complete data flow for the project:

1. **Static Covariates (Fixed for each BA):**  
   * Sensitivity coefficients (Gas, Hydro, Solar) derived from EIA-930 historical fuel mix.7  
   * State-to-BA mapping weights for multi-state regions.31  
2. **Time-Varying Known Inputs (Known for the future):**  
   * Calendar features (Day of week, season, holidays).  
   * Weather forecasts (Temperature, humidity, solar irradiance).  
3. **Time-Varying Observed Inputs (Historical sentiment):**  
   * GDELT AvgTone (Regionalized via FIPS/Coordinates).16  
   * State-level Daily EPU Index.23  
   * Categorical Uncertainty (Trade, Regulatory, Energy).20  
4. **TFT Output (Multi-horizon):**  
   * Probabilistic demand forecasts for the next 24 to 168 hours, with interpretable attention scores identifying which sentiment spikes drove the prediction.3

This methodology ensures that the forecasting model is not only accurate under normal conditions but remains robust and adaptive during periods of global instability, providing the "crisis-resilience" required for modern energy management.

#### **Works cited**

1. Causal-Aware Multimodal Transformer for Supply Chain Demand Forecasting: Integrating Text, Time Series, and Satellite Imagery \- IEEE Xplore, accessed on March 29, 2026, [https://ieeexplore.ieee.org/iel8/6287639/10820123/11197533.pdf](https://ieeexplore.ieee.org/iel8/6287639/10820123/11197533.pdf)  
2. From Market Volatility to Predictive Insight: An Adaptive Transformer–RL Framework for Sentiment-Driven Financial Time-Series Forecasting \- MDPI, accessed on March 29, 2026, [https://www.mdpi.com/2571-9394/7/4/55](https://www.mdpi.com/2571-9394/7/4/55)  
3. Forecasting Energy Consumption Demand of Customers in Smart Grid Using Temporal Fusion Transformer (TFT) \- ResearchGate, accessed on March 29, 2026, [https://www.researchgate.net/publication/367323447\_Forecasting\_Energy\_Consumption\_Demand\_of\_Customers\_in\_Smart\_Grid\_Using\_Temporal\_Fusion\_Transformer\_TFT](https://www.researchgate.net/publication/367323447_Forecasting_Energy_Consumption_Demand_of_Customers_in_Smart_Grid_Using_Temporal_Fusion_Transformer_TFT)  
4. The asymmetric effect of economic policy uncertainty on energy ..., accessed on March 29, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC9065669/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9065669/)  
5. Temporal Fusion Transformer-Based Trading Strategy for Multi-Crypto Assets Using On-Chain and Technical Indicators \- MDPI, accessed on March 29, 2026, [https://www.mdpi.com/2079-8954/13/6/474](https://www.mdpi.com/2079-8954/13/6/474)  
6. (PDF) Temporal Fusion Transformer-Based Trading Strategy for Multi-Crypto Assets Using On-Chain and Technical Indicators \- ResearchGate, accessed on March 29, 2026, [https://www.researchgate.net/publication/392749720\_Temporal\_Fusion\_Transformer-Based\_Trading\_Strategy\_for\_Multi-Crypto\_Assets\_Using\_On-Chain\_and\_Technical\_Indicators](https://www.researchgate.net/publication/392749720_Temporal_Fusion_Transformer-Based_Trading_Strategy_for_Multi-Crypto_Assets_Using_On-Chain_and_Technical_Indicators)  
7. Fuel Mix \- ERCOT.com, accessed on March 29, 2026, [https://www.ercot.com/gridmktinfo/dashboards/fuelmix](https://www.ercot.com/gridmktinfo/dashboards/fuelmix)  
8. Sensitivity Analyses and Sensitivity Coefficients of Standardized Daily ASCE-Penman-Monteith Equation | Request PDF \- ResearchGate, accessed on March 29, 2026, [https://www.researchgate.net/publication/228341883\_Sensitivity\_Analyses\_and\_Sensitivity\_Coefficients\_of\_Standardized\_Daily\_ASCE-Penman-Monteith\_Equation](https://www.researchgate.net/publication/228341883_Sensitivity_Analyses_and_Sensitivity_Coefficients_of_Standardized_Daily_ASCE-Penman-Monteith_Equation)  
9. Beyond Polarity: Multi-Dimensional LLM Sentiment Signals for WTI Crude Oil Futures Return Prediction \- arXiv, accessed on March 29, 2026, [https://arxiv.org/html/2603.11408v1](https://arxiv.org/html/2603.11408v1)  
10. (PDF) A hybrid prophet-based framework for multimodal forecasting with market sentiment signals \- ResearchGate, accessed on March 29, 2026, [https://www.researchgate.net/publication/403074638\_A\_hybrid\_prophet-based\_framework\_for\_multimodal\_forecasting\_with\_market\_sentiment\_signals](https://www.researchgate.net/publication/403074638_A_hybrid_prophet-based_framework_for_multimodal_forecasting_with_market_sentiment_signals)  
11. Dynamic Spillovers of Economic Policy Uncertainty: A TVP-VAR Analysis of Latin American and Global EPU Indices \- MDPI, accessed on March 29, 2026, [https://www.mdpi.com/2227-7099/13/1/11](https://www.mdpi.com/2227-7099/13/1/11)  
12. A Multi-Modal Approach Using a Hybrid Vision Transformer and Temporal Fusion Transformer Model for Stock Price Movement Classifi \- IEEE Xplore, accessed on March 29, 2026, [https://ieeexplore.ieee.org/iel8/6287639/10820123/11080418.pdf](https://ieeexplore.ieee.org/iel8/6287639/10820123/11080418.pdf)  
13. Advancing electric demand forecasting through the temporal fusion transformer model | Request PDF \- ResearchGate, accessed on March 29, 2026, [https://www.researchgate.net/publication/379212010\_Advancing\_electric\_demand\_forecasting\_through\_the\_temporal\_fusion\_transformer\_model](https://www.researchgate.net/publication/379212010_Advancing_electric_demand_forecasting_through_the_temporal_fusion_transformer_model)  
14. Forecasting, Volume 7, Issue 4 (December 2025\) – 27 articles \- MDPI, accessed on March 29, 2026, [https://www.mdpi.com/2571-9394/7/4](https://www.mdpi.com/2571-9394/7/4)  
15. Forecasting Commodity Price Shocks Using Temporal and Semantic Fusion of Prices Signals and Agentic Generative AI Extracted Economic News \- arXiv, accessed on March 29, 2026, [https://arxiv.org/html/2508.06497v1](https://arxiv.org/html/2508.06497v1)  
16. The GDELT Project, accessed on March 29, 2026, [https://www.gdeltproject.org/](https://www.gdeltproject.org/)  
17. Big Data Analytics Dataset List \- Electrical Engineering \- Columbia University, accessed on March 29, 2026, [https://www.ee.columbia.edu/\~cylin/course/bigdata/getdatasetinfo.html](https://www.ee.columbia.edu/~cylin/course/bigdata/getdatasetinfo.html)  
18. The GDELT Project \- Kaggle, accessed on March 29, 2026, [https://www.kaggle.com/gdelt/gdelt/metadata](https://www.kaggle.com/gdelt/gdelt/metadata)  
19. Free Access to World News: Reconstructing Full-Text Articles from GDELT \- MDPI, accessed on March 29, 2026, [https://www.mdpi.com/2504-2289/10/2/45](https://www.mdpi.com/2504-2289/10/2/45)  
20. US EPU (Monthly, Daily, Categorical), accessed on March 29, 2026, [https://www.policyuncertainty.com/us\_monthly.html](https://www.policyuncertainty.com/us_monthly.html)  
21. ECONOMIC POLICY UNCERTAINTY AND MACROECONOMY: A MULTI-COUNTRY FRAMEWORK \- International Journal Of Business & Management Studies, accessed on March 29, 2026, [https://ijbms.net/assets/files/1758594847.pdf](https://ijbms.net/assets/files/1758594847.pdf)  
22. Economic Policy Uncertainty Index for United States (USEPUINDXD) | FRED | St. Louis Fed, accessed on March 29, 2026, [https://fred.stlouisfed.org/series/USEPUINDXD](https://fred.stlouisfed.org/series/USEPUINDXD)  
23. Economic Policy Uncertainty Index for United States (USEPUINDXM) | FRED | St. Louis Fed, accessed on March 29, 2026, [https://fred.stlouisfed.org/series/USEPUINDXM](https://fred.stlouisfed.org/series/USEPUINDXM)  
24. Sentiment and Volatility in Financial Markets: A Review of BERT and GARCH Applications during Geopolitical Crises \- arXiv, accessed on March 29, 2026, [https://arxiv.org/html/2510.16503v1](https://arxiv.org/html/2510.16503v1)  
25. WORKING PAPER \- Economic Policy Uncertainty Index, accessed on March 29, 2026, [http://www.policyuncertainty.com/media/OHouari%20(2025)%20MoroccanEPU.pdf](http://www.policyuncertainty.com/media/OHouari%20\(2025\)%20MoroccanEPU.pdf)  
26. Economic Policy Uncertainty (EPU) sentiment index by countries & media source | Geopolitics \- BBVA Research, accessed on March 29, 2026, [https://bigdata.bbvaresearch.com/en/geopolitics/geopolitics-economics/economic-policy-uncertainty/countries/](https://bigdata.bbvaresearch.com/en/geopolitics/geopolitics-economics/economic-policy-uncertainty/countries/)  
27. US \- Economic Policy Uncertainty Index | MacroMicro, accessed on March 29, 2026, [https://en.macromicro.me/charts/26279/us-epu](https://en.macromicro.me/charts/26279/us-epu)  
28. Opendata \- U.S. Energy Information Administration (EIA), accessed on March 29, 2026, [https://www.eia.gov/opendata/](https://www.eia.gov/opendata/)  
29. Hourly Balancing Authority Operations Report \- ERCOT.com, accessed on March 29, 2026, [https://www.ercot.com/mp/data-products/data-product-details?id=EIA-930-CD](https://www.ercot.com/mp/data-products/data-product-details?id=EIA-930-CD)  
30. Generation \- ERCOT.com, accessed on March 29, 2026, [https://www.ercot.com/gridinfo/generation](https://www.ercot.com/gridinfo/generation)  
31. daily electricity generation mix \- Real-time Operating Grid \- U.S. Energy Information Administration (EIA), accessed on March 29, 2026, [https://www.eia.gov/electricity/gridmonitor/dashboard/daily\_generation\_mix/US48/US48](https://www.eia.gov/electricity/gridmonitor/dashboard/daily_generation_mix/US48/US48)  
32. Effects of climate change on the power system: a case study of the southeast U.S. Submitted in partial fulfillment of the requir, accessed on March 29, 2026, [https://www.cmu.edu/ceic/research-publications/francisco-ralston-fonseca-phd-thesis-2020.pdf](https://www.cmu.edu/ceic/research-publications/francisco-ralston-fonseca-phd-thesis-2020.pdf)  
33. I. Sustaining stability amid uncertainty and fragmentation \- Bank for International Settlements, accessed on March 29, 2026, [https://www.bis.org/publ/arpdf/ar2025e1.htm](https://www.bis.org/publ/arpdf/ar2025e1.htm)  
34. A hybrid transformer framework integrating sentiment and dynamic market structure for stock price movement forecasting | Request PDF \- ResearchGate, accessed on March 29, 2026, [https://www.researchgate.net/publication/400224022\_A\_hybrid\_transformer\_framework\_integrating\_sentiment\_and\_dynamic\_market\_structure\_for\_stock\_price\_movement\_forecasting](https://www.researchgate.net/publication/400224022_A_hybrid_transformer_framework_integrating_sentiment_and_dynamic_market_structure_for_stock_price_movement_forecasting)  
35. Energies, Volume 17, Issue 7 (April-1 2024\) – 282 articles, accessed on March 29, 2026, [https://www.mdpi.com/1996-1073/17/7](https://www.mdpi.com/1996-1073/17/7)

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAZCAYAAAAIcL+IAAAAuklEQVR4Xu2QOwoCQRBES1QQNDMQP5Gmm5kaiIh4IxMz8Uhi4gU8hGC2oZGR+KliepfeHTEyEXzwgqnp7e5Z4M83adIFrdu5Skd0ShuWoUI3dvmkezqzfEkftK3CPk0QuqrwotAY0yvt6pC1ntOUDu0sVggfq0khPJTCHUJhjroq3PqQnFEq1Fgt7ccKFWlKTrZLy2U1ywpTjvTuAzJB/Dj06MAHeP+4iA49Iez+EY29IX5cxBrhL/wUL1haH7vbDY8LAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAaCAYAAACKER0bAAAAg0lEQVR4XmNgGAW4ACsQhwCxGboECHQD8S8g3gHED4DYE1lSE4i/ArEflO8NxCdgkruA+D8Qs0D5wkB8GoiDYAr+QTFWIM4A0b0GXQIGJBkgCqrQJWAAZC9IQTmaOMi7B2Gc90B8CyHHYAXEp4BYAiZgD8R3gPgJED+CssNgkqOAgQEAj74X6pyhxtYAAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,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>
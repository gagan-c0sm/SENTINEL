# Research Paper Figure Specification: SENTINEL Architecture Diagram

## 1. Objective & Context
This document serves as a referential guide for visual designers and technical illustrators to create **Figure 1 (Main Architecture)** for our upcoming academic conference submission. 

The reference image provided shows a generic time-series structure. We must adapt this into a **formal, publication-ready schematic** detailing the Temporal Fusion Transformer (TFT) infused with Global Knowledge Graph (GKG) signals. 

**The Academic Goal**: The diagram must immediately convey our novel contribution to the reviewers: the structural integration of **macro-geopolitical risk (GPR)** and **regional grid sentiment** as observed covariates in an attention-based energy forecasting model, differentiating it from traditional weather-only architectures.

---

## 2. Overall Layout & Mathematical Structure
The diagram should be divided into two main temporal zones by a vertical dashed line, utilizing standard time-series notation:

*   **Left Side (Encoder Layer)**: Represents historical data sequence. Annotated as $t \in [t_{0} - k, t_{0})$, where $k = 168$.
*   **Vertical Division**: The present forecast baseline, annotated as $t_{0}$.
*   **Right Side (Decoder Layer)**: Represents the forecasting horizon. Annotated as $t \in [t_{0}, t_{0} + \tau]$, where $\tau = 24$.

Data flows bottom-up: from raw feature matrices $\mathbf{X}^{(past)}$ and $\mathbf{X}^{(known)}$, through the TFT embedding layers, and outputting the quantile forecast $\hat{\mathbf{y}}$.

---

## 3. The Components (Nodes & Layers)

Here are the specific, localized blocks that must be present, mapped to our exact feature space.

### A. Static Covariates ($\mathbf{s}$)
*   **Placement**: Middle-right or bottom-left, feeding directly into the TFT block across all time steps.
*   **Features to List**: 
    *   `ba_code` (Balancing Authority Identity)
    *   Fuel Mix Sensitivities (Gas, Renewable, Nuclear)

### B. Known Future Inputs ($\mathbf{X}^{(known)}$)
*   **Placement**: A wide base layer spanning *across* the $t_{0}$ line, continuing into the decoder. These are deterministic or highly reliable forecasts.
*   **Features to List**:
    *   **Temporal/Calendar**: Time of day, day of week, holiday markers.
    *   **Meteorological Forecasts**: $\text{Temp}$, $\text{HDD}$, $\text{CDD}$.
    *   **Extrapolated Baselines**: Prophet (Trend, Seasonality).

### C. Observed Historical Inputs ($\mathbf{X}^{(past)}$) — *Our Main Contribution*
*   **Placement**: A distinct block sitting on top of the 'Known Inputs', strictly residing in the **Encoder Layer (Left Side)**. 
*   **Design Note**: **This module must be visually emphasized (e.g., bolded outline or distinct shading) as it contains the primary research novelty.**
*   **Features to List**:
    *   **Grid Physicals**: Supply margin, real-time generation mix.
    *   **Geopolitical & Sentiment Vectors (GKG)**: 
        *   Global Geopolitical Risk ($GPR_{z}$)
        *   Regional Grid Stress ($Stress_{z}$)
        *   Infrastructure Sentiment ($Pipeline_{z}$)

### D. Past Targets ($y_{t}$)
*   **Placement**: A continuous curve in the top-left quadrant portraying historical energy demand.
*   **Label**: Historical Demand $y_{t}$
*   **Visual detail**: Vertical mapping arrows linking the $t-1, t-2$ sequence embeddings directly to points on this curve.

### E. The Core Processing Engine: Temporal Fusion Transformer (TFT)
*   The central node straddling $t_0$. It should not be a black box; we must expose the key internal mechanisms responsible for our results.
*   **Internal Sub-blocks to explicitly show**:
    1.  **Variable Selection Network (VSN)**: Visually represent this acting on the GKG inputs, emphasizing that it learns dynamic, instance-wise weights (acting as the regime-switching mechanism during crises).
    2.  **LSTM Seq2Seq Encoder-Decoder**: The recurrent memory backbone.
    3.  **Multi-Head Attention**: The layer that scans the LSTM outputs to establish long-term dependencies (e.g., connecting a past geopolitical spike to a future demand shock).

### F. Outputs: Quantile Forecasts ($\hat{y}_{t}$)
*   **Placement**: Top-Right quadrant.
*   **Visual detail**: 
    *   A dashed projection curve originating from the Past Target line at $t_0$.
    *   Vertical bars at distinct forecast horizons ($t_0+1, t_0+12, \dots$) showing the upper and lower bounds.
    *   **Labels**: Point Forecast ($\hat{y}(q=0.5)$), Prediction Intervals ($\hat{y}(q=0.1)$ to $\hat{y}(q=0.9)$).

---

## 4. Connections & Data Flows (The Edges)
1.  **Bottom-Up Wiring**: Feed-forward arrows from $\mathbf{X}^{(past)}$ and $\mathbf{X}^{(known)}$ into the VSN layer of the TFT.
2.  **Temporal Severance**: The *Observed Inputs* flow strictly stops at the $t_0$ line. The *Known Inputs* flow continues into the right side of the diagram, feeding the decoder.
3.  **Forecast Projection**: From the overarching Attention block, a definitive output arrow projects upward to generate the future quantiles.

---

## 5. Academic Style & Formatting Guidelines
Since this is for a conference paper (typically two-column IEEE or NeurIPS format), the diagram must be highly legible when scaled down.

### Format & Colors (Print-Ready Academic Style)
*   **Format**: Must be delivered as a scalable vector graphic (**PDF, EPS, or SVG**). No raster PNGs for the final submission.
*   **Background**: Strict White or Transparent background. (Dark mode is generally not accepted for leading ML/Energy conference papers).
*   **Color Palette (Colorblind Safe & Print Safe)**:
    *   Lines & Structural Boxes: Charcoal Grey (`#333333`) or Black.
    *   **Standard Inputs** (Weather/Time): Muted Slate Blue or Soft Grey.
    *   **Novel GKG Inputs**: A highly contrasting, distinct color like Crimson (`#D32F2F`) or Deep Teal (`#00796B`) to draw the reviewer's eye immediately.
    *   **Past Target Line**: Solid dark line.
    *   **Future Prediction Curve**: Dashed line with semi-transparent shading for the quantile intervals (e.g., light blue fill with alpha=0.3).

### Typography
*   Use a serif font for mathematical notation (e.g., Times New Roman or Computer Modern to match LaTeX output) and a clean sans-serif (e.g., Arial or Helvetica) for block labels.
*   Ensure all subscripts ($t_0$) and text are large enough to be readable when the figure is shrunk to a 3.5-inch column width.

### Suggested Figure Caption (For the Paper)
*The designer should keep this caption in mind to ensure the visual narrative aligns with the text:*
> **Figure 1**: Architecture of the SENTINEL forecasting model. The model utilizes a Temporal Fusion Transformer (TFT) backend. Past targets ($y_t$) and time-varying observed inputs—crucially including our novel integration of GDELT-derived Global Geopolitical Risk ($GPR_z$) and regional infrastructure sentiment indicators—are processed by the encoder for $t < t_0$. Time-varying known inputs (e.g., meteorological forecasts and temporal markers) are provided to both the encoder and decoder. Variable Selection Networks (VSNs) dynamically regulate the importance of the geopolitical signals during identified crisis regimes. The model outputs multi-horizon conditional quantile forecasts ($\hat{y}_t$) to quantify predictive uncertainty.

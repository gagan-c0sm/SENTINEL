# SENTINEL: Decentralized Oracle Network (DON) Architecture

This document outlines the foundation for **Phase 5: Blockchain Deployment**. It explains how off-chain ML predictions are moved on-chain, and why cryptoeconomic staking is strictly required for grid accountability.

---

## 1. How the DON Operates Technically

To bridge our local Python machine learning models to the blockchain, we will build a 3-part pipeline:

### A. The Off-Chain Model (Python)
Each team member (Node A, Node B, Node C) runs the identical PyTorch `.pt` model locally. 
Every 24 hours, the model generates a JSON payload:
```json
{
  "BA": "ERCO",
  "TargetDate": "2026-03-25T12:00:00Z",
  "Predicted_MW": 74500,
  "Supply_Disruption_Prob": 0.92
}
```

### B. The Oracle Bridge (`Web3.py`)
Because Ethereum smart contracts cannot run Python code or fetch internet data (they exist in a closed environment), we use the `web3.py` library to push the data *into* the blockchain.
1. The Python script takes the JSON prediction.
2. It cryptographically signs the data with the node's **Private Key**.
3. It pays a microscopic block fee (gas) to broadcast a `submitPrediction()` transaction to the network.

### C. The On-Chain Smart Contract (Solidity)
We will deploy a `SentinelConsensus.sol` smart contract on a free testnet (like Ethereum Sepolia or Polygon Amoy).
1. The contract receives predictions from Node A, Node B, and Node C.
2. It checks to ensure the signatures match registered, authorized Oracle nodes.
3. It takes the **Median** of all predictions to filter out any single computer that crashed or was hacked.
4. It permanently logs the final consensus prediction to the immutable public ledger.

---

## 2. Why is Cryptocurrency Staking Proposed?

If SENTINEL's predictions are used to actually trigger physical grid responses (like buying 5,000 Megawatts of emergency reserve power), whoever controls the Oracle predictions effectively controls the grid.

**The Sybil Attack Problem:**
If there is no financial cost to submitting a prediction, a malicious actor (or foreign nation) could quickly spin up 10,000 fake Oracle nodes. They would outvote Node A, B, and C, forcing the Smart Contract to accept a fake prediction (e.g., "The Texas grid is perfectly fine tonight"). The grid operators would ignore the physical warnings, resulting in blackouts.

**The Solution: Cryptoeconomic Staking & Slashing**
To be legally allowed to submit a prediction to the SENTINEL Smart Contract, a node must **"Stake"** (lock up) cryptocurrency, for instance, $10,000 worth of Ethereum.

1. **Skin in the Game:** To spin up 10,000 fake nodes, the hacker would now need $100 Million in real capital, making manipulation financially unviable.
2. **Slashing (The Penalty):** 48 hours after the prediction, the Smart Contract connects to the official EIA after-the-fact dataset. If Node B predicted 50,000 MW, but the actual reality was 100,000 MW (meaning their model was horribly broken), the smart contract mathematically seizes their staked $10,000. This is called **Slashing**. 
3. **Reward:** The nodes that predicted correctly (Node A and Node C) split the slashed penalty as a reward for providing highly accurate data.

### Conclusion
Cryptocurrency staking isn't about traditional "trading" or "speculation". It is applied **Game Theory**. It provides a mathematical, physics-based guarantee that only highly accurate, honest Machine Learning models will ever submit grid data. If you upload a bad ML model, you lose real money.

---

## 3. The Ultimate Goal: Automated Supply & Demand Balancing
The ultimate reason for bringing SENTINEL's predictions on-chain via the DON is so that **Smart Contracts can automatically execute physical grid balancing**.

Instead of human ISO operators calling each other on the phone during an escalating crisis, a smart contract completely automates the macro-grid:
1. **The Oracle Trigger:** The DON pushes an official, staked prediction: *"ERCOT (Texas) will face a 10,000 Megawatt deficit tomorrow at 3 PM due to a natural gas failure."*
2. **The Matching Engine:** A `GridBalancer.sol` smart contract reads the DON. It instantly checks the predictions for the surrounding states. It sees that the Southwest Power Pool (SWPP) is predicted to have a 12,000 MW surplus of wind energy at the exact same time.
3. **The Automated Trade:** The smart contract instantly executes a cryptographically binding agreement: *ERCOT purchases 10,000 MW from SWPP for tomorrow at 3 PM, and the transmission path is reserved.*

By combining AI forecasting with decentralized Smart Contracts, we eliminate the slow, bureaucratic reaction times that typically cause cascading blackouts. The entire electrical grid becomes an automated, self-healing mathematical network.

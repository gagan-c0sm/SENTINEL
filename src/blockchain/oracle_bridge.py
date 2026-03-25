import os
import json
import time
import argparse
from web3 import Web3
from web3.middleware import construct_sign_and_send_raw_middleware
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# =========================================================================
# SENTINEL Oracle Configuration 
# Connects the local Python ML Environment to the global decentralized network
# =========================================================================

RPC_URL = os.getenv("RPC_URL", "https://sepolia.infura.io/v3/YOUR-PROJECT-ID")
PRIVATE_KEY = os.getenv("ORACLE_PRIVATE_KEY", "0x0000000000000000000000000000000000000000000000000000000000000000")
CONTRACT_ADDRESS = os.getenv("SENTINEL_CONTRACT_ADDRESS", "0x0000000000000000000000000000000000000000")

def load_abi():
    """Loads the compiled SentinelConsensus.sol ABI JSON"""
    abi_path = os.path.join(os.path.dirname(__file__), "artifacts", "SentinelConsensus.json")
    if os.path.exists(abi_path):
        with open(abi_path, 'r') as f:
            return json.load(f)['abi']
    else:
        # Fallback ABI purely for the submitPrediction function signature matching
        return [
            {
                "inputs": [
                    {"internalType": "string", "name": "baCode", "type": "string"},
                    {"internalType": "uint256", "name": "targetDate", "type": "uint256"},
                    {"internalType": "uint256", "name": "predictedMW", "type": "uint256"},
                    {"internalType": "uint256", "name": "disruptionProb", "type": "uint256"}
                ],
                "name": "submitPrediction",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]

def push_prediction_to_chain(ba_code: str, target_date_unix: int, predicted_mw: int, disruption_prob: float):
    """
    Cryptographically signs the Python Machine Learning forecast and broadcasts it 
    into the Ethereum Smart Contract for consensus processing.
    """
    logger.info(f"Connecting to Blockchain RPC: {RPC_URL}")
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    
    if not w3.is_connected():
        logger.error("SENTINEL Oracle Error: Failed to connect to Ethereum network!")
        return False
        
    try:
        # Inject the Private Key into the node's middleware
        account = w3.eth.account.from_key(PRIVATE_KEY)
        w3.middleware_onion.add(construct_sign_and_send_raw_middleware(account))
        logger.info(f"Oracle Node securely authenticated as: {account.address}")
        
        contract = w3.eth.contract(address=w3.to_checksum_address(CONTRACT_ADDRESS), abi=load_abi())
        
        # Scale probability safely: 0.9250 -> 9250 (Smart Contracts don't support floats natively)
        scaled_prob = int(disruption_prob * 10000)
        
        logger.info(f"Broadcasting Forecast for {ba_code} at Unix {target_date_unix} | Prediction: {predicted_mw} MW | Risk: {scaled_prob}/10000")
        
        # Build the exact blockchain Transaction payload
        tx = contract.functions.submitPrediction(
            baCode=ba_code,
            targetDate=target_date_unix,
            predictedMW=predicted_mw,
            disruptionProb=scaled_prob
        ).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gas': 2000000,
            'gasPrice': w3.to_wei('20', 'gwei')
        })
        
        # Sign the payload with mathematical cryptographic keys
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
        
        # Actively push the finalized payload to the global Mempool
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        logger.success(f"Transaction successfully broadcasted! Hash: {w3.to_hex(tx_hash)}")
        
        # Await miner processing
        logger.info("Waiting for block mining confirmation...")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if receipt.status == 1:
            logger.success(f"Consensus prediction permanently secured in Blockchain Block #{receipt.blockNumber}!")
            return True
        else:
            logger.error("Transaction natively reverted! Reason: Node isn't registered, no active ETH stake, or maliciously formatted parameters.")
            return False
            
    except Exception as e:
        logger.error(f"Blockchain bridge failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SENTINEL Decentralized Oracle Network Bridge")
    parser.add_argument("--ba", type=str, required=True, help="Balancing Authority Code (e.g., ERCO)")
    parser.add_argument("--mw", type=int, required=True, help="Predicted Megawatts")
    parser.add_argument("--risk", type=float, required=True, help="Geopolitical/Weather Risk Probability (0.0 - 1.0)")
    args = parser.parse_args()
    
    # Calculate a forecast for exactly 24 hours into the future
    target_time = int(time.time()) + (24 * 3600)
    
    push_prediction_to_chain(args.ba, target_time, args.mw, args.risk)

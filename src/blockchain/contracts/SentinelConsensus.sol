// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title SentinelConsensus
 * @dev Decentralized Oracle Network (DON) for Energy Grid Predictions
 * 
 * Implements strict cryptoeconomic staking to prevent Sybil attacks.
 * Off-chain ML models must stake ETH to submit grid demand forecasts.
 */
contract SentinelConsensus {
    address public admin;
    uint256 public constant REQUIRED_STAKE = 1 ether; 

    struct Prediction {
        address nodeId;
        string baCode;
        uint256 targetDate; // Unix timestamp for the forecasted hour
        uint256 predictedMW;
        uint256 disruptionProb; // Scaled by 10000 (e.g., 9200 = 92.00%)
    }

    struct Node {
        bool isRegistered;
        uint256 stakedBalance;
        uint256 totalPredictions;
        uint256 slashedCount;
    }

    mapping(address => Node) public oracleNodes;
    
    // Hash of (baCode + targetDate) -> Array of submitted predictions
    mapping(bytes32 => Prediction[]) public predictionRounds;
    
    // Events for dashboard/webhook monitoring
    event NodeRegistered(address indexed node, uint256 stake);
    event NodeSlashed(address indexed node, uint256 slashAmount, string reason);
    event PredictionSubmitted(
        address indexed node, 
        bytes32 indexed roundId, 
        string baCode, 
        uint256 predictedMW
    );
    event ConsensusFinalized(
        bytes32 indexed roundId, 
        string baCode, 
        uint256 targetDate, 
        uint256 medianMW, 
        uint256 medianProb
    );

    constructor() {
        admin = msg.sender;
    }

    modifier onlyAdmin() {
        require(msg.sender == admin, "SENTINEL: Only admin can execute");
        _;
    }

    modifier onlyRegisteredNode() {
        require(oracleNodes[msg.sender].isRegistered, "SENTINEL: Unauthorized un-registered node");
        require(oracleNodes[msg.sender].stakedBalance >= REQUIRED_STAKE, "SENTINEL: Insufficient active stake");
        _;
    }

    /**
     * @dev Nodes must legally lock up funds to submit predictions (Game Theory alignment)
     */
    function registerNode() external payable {
        require(!oracleNodes[msg.sender].isRegistered, "SENTINEL: Node already registered");
        require(msg.value >= REQUIRED_STAKE, "SENTINEL: Must stake minimum required ETH");

        oracleNodes[msg.sender] = Node({
            isRegistered: true,
            stakedBalance: msg.value,
            totalPredictions: 0,
            slashedCount: 0
        });

        emit NodeRegistered(msg.sender, msg.value);
    }

    /**
     * @dev Called via Web3.py by the off-chain Python models
     */
    function submitPrediction(
        string memory baCode,
        uint256 targetDate,
        uint256 predictedMW,
        uint256 disruptionProb
    ) external onlyRegisteredNode {
        require(disruptionProb <= 10000, "SENTINEL: Probability must be 0-10000");

        // Unique cryptographic identifier for this specific grid hour
        bytes32 roundId = keccak256(abi.encodePacked(baCode, targetDate));

        Prediction memory newPred = Prediction({
            nodeId: msg.sender,
            baCode: baCode,
            targetDate: targetDate,
            predictedMW: predictedMW,
            disruptionProb: disruptionProb
        });

        predictionRounds[roundId].push(newPred);
        oracleNodes[msg.sender].totalPredictions++;

        emit PredictionSubmitted(msg.sender, roundId, baCode, predictedMW);
    }

    /**
     * @dev Executes mathematical median sorting to finalize the official truth.
     * Drops severe outliers (e.g. hacked nodes).
     */
    function finalizeConsensus(bytes32 roundId) external onlyAdmin {
        Prediction[] memory roundPreds = predictionRounds[roundId];
        require(roundPreds.length > 0, "SENTINEL: No predictions for this round");

        uint256 l = roundPreds.length;
        uint256[] memory mwValues = new uint256[](l);
        uint256[] memory probValues = new uint256[](l);

        for(uint i = 0; i < l; i++) {
            mwValues[i] = roundPreds[i].predictedMW;
            probValues[i] = roundPreds[i].disruptionProb;
        }

        uint256 medianMW = _findMedian(mwValues);
        uint256 medianProb = _findMedian(probValues);

        // Store or broadcast the officially agreed upon Sentinel prediction
        emit ConsensusFinalized(roundId, roundPreds[0].baCode, roundPreds[0].targetDate, medianMW, medianProb);
    }

    // Standard bubble sort for small arrays (3-15 nodes)
    function _findMedian(uint256[] memory arr) internal pure returns (uint256) {
        uint256 l = arr.length;
        for(uint i = 0; i < l; i++) {
            for(uint j = i+1; j < l; j++) {
                if(arr[i] > arr[j]) {
                    uint256 temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
        }
        if (l % 2 == 0) {
            return (arr[l/2 - 1] + arr[l/2]) / 2;
        } else {
            return arr[l/2];
        }
    }

    /**
     * @dev Slash (seize) funds from a node if reality proves their prediction was maliciously wrong
     */
    function slashNode(address badNode, uint256 slashAmount, string memory reason) external onlyAdmin {
        require(oracleNodes[badNode].isRegistered, "SENTINEL: Node not registered");
        require(oracleNodes[badNode].stakedBalance >= slashAmount, "SENTINEL: Slash exceeds active balance");
        
        oracleNodes[badNode].stakedBalance -= slashAmount;
        oracleNodes[badNode].slashedCount++;
        
        emit NodeSlashed(badNode, slashAmount, reason);
    }
}

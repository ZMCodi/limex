#!/bin/bash
# Script to launch multiple trading bots in separate terminals

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate assets

# Total capital
TOTAL_CAPITAL=90000

# Define symbols to trade
SYMBOLS=("AAPL" "LLY" "COST" "BRK.B" "CAT" "OKLO" "LIN" "AMT" "AWK")

# Calculate allocation per symbol
NUM_SYMBOLS=${#SYMBOLS[@]}
ALLOCATION=$(echo "$TOTAL_CAPITAL / $NUM_SYMBOLS" | bc -l)

echo "Starting trading system with \$$TOTAL_CAPITAL capital"
echo "Trading $NUM_SYMBOLS symbols: ${SYMBOLS[*]}"
echo "Capital per symbol: \$$ALLOCATION"

# Determine OS and use appropriate terminal command
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    for symbol in "${SYMBOLS[@]}"; do
        gnome-terminal --title="$symbol Bot" -- bash -c "conda activate assets && python3 main.py \"$symbol\" \"$ALLOCATION\""
    done
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    for symbol in "${SYMBOLS[@]}"; do
        osascript -e "tell app \"Terminal\" to do script \"cd $(pwd) && conda activate assets && python3 main.py $symbol $ALLOCATION\""
    done
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    for symbol in "${SYMBOLS[@]}"; do
        start cmd /k "conda activate assets && python main.py \"$symbol\" \"$ALLOCATION\""
    done
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "All bots have been launched in separate terminals"

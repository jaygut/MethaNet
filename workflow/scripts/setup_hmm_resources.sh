#!/bin/bash
set -e

# MethaNet HMM Resource Setup
# This script downloads the TIGRFAMs library and extracts the "Strategic Fix" marker set
# required for robust methane flux prediction in coastal/mangrove ecosystems.
#
# Usage: ./workflow/scripts/setup_hmm_resources.sh
# Run this from the repository root.

# Ensure data directory exists
mkdir -p data/hmm
cd data/hmm

echo "==> Setting up HMM resources in $(pwd)..."

# Download TIGRFAMs library (v15.0)
if [ ! -f "TIGRFAMs_15.0_HMM.LIB" ]; then
    echo "Downloading TIGRFAMs library (approx 106 MB)..."
    # Use wget with no-clobber, or curl if wget missing
    if command -v wget &> /dev/null; then
        wget -nc https://ftp.ncbi.nlm.nih.gov/hmm/TIGRFAMs/release_15.0/TIGRFAMs_15.0_HMM.LIB.gz
    else
        curl -O https://ftp.ncbi.nlm.nih.gov/hmm/TIGRFAMs/release_15.0/TIGRFAMs_15.0_HMM.LIB.gz
    fi
    
    echo "Decompressing library..."
    gunzip -f TIGRFAMs_15.0_HMM.LIB.gz
else
    echo "TIGRFAMs library already present."
fi

# Index for fast retrieval (requires HMMER)
if [ ! -f "TIGRFAMs_15.0_HMM.LIB.ssi" ]; then
    if command -v hmmfetch &> /dev/null; then
        echo "Indexing HMM library..."
        hmmfetch --index TIGRFAMs_15.0_HMM.LIB
    else
        echo "WARNING: hmmfetch not found. Skipping indexing. Ensure HMMER is installed."
    fi
fi

# Cache a list of available accessions/names for debugging and fallback lookups.
# This is also useful on HPC where some TIGR accessions differ across releases.
if [ -f "TIGRFAMs_15.0_HMM.LIB.ssi" ] && [ ! -f "TIGRFAMs_15.0_HMM.LIB.keys" ]; then
    echo "Caching TIGRFAM keys (accession + name)..."
    hmmfetch -l TIGRFAMs_15.0_HMM.LIB > TIGRFAMs_15.0_HMM.LIB.keys
fi

echo "Extracting marker profiles..."

MISSING_MARKERS=()

# Function to fetch HMM if not exists
fetch_hmm() {
    local outfile=$1
    local accession=$2
    local name=$3
    local fallback_patterns=${4:-}
    
    if [ ! -f "$outfile" ]; then
        echo "  - Extracting $name ($accession) -> $outfile"
        if hmmfetch -o "$outfile" TIGRFAMs_15.0_HMM.LIB "$accession" 2>/dev/null; then
            return 0
        fi

        echo "WARNING: $accession not found in TIGRFAMs_15.0_HMM.LIB. Attempting fallback lookup..."
        if [ -z "$fallback_patterns" ]; then
            fallback_patterns="$name"
        fi

        if [ ! -f "TIGRFAMs_15.0_HMM.LIB.keys" ]; then
            hmmfetch -l TIGRFAMs_15.0_HMM.LIB > TIGRFAMs_15.0_HMM.LIB.keys
        fi

        # keys file format is: <accession> <name>
        local resolved_acc=""
        local pat
        IFS='|' read -r -a pat_arr <<< "$fallback_patterns"
        for pat in "${pat_arr[@]}"; do
            pat=$(echo "$pat" | xargs)
            if [ -z "$pat" ]; then
                continue
            fi
            resolved_acc=$(awk -v pat="$pat" 'BEGIN{IGNORECASE=1} $0 ~ pat {print $1; exit}' TIGRFAMs_15.0_HMM.LIB.keys)
            if [ -n "$resolved_acc" ]; then
                echo "  - Fallback matched accession $resolved_acc for '$name' (pattern: $pat)"
                hmmfetch -o "$outfile" TIGRFAMs_15.0_HMM.LIB "$resolved_acc"
                return 0
            fi
        done

        echo "ERROR: Could not resolve an accession for '$name' using fallback patterns: '$fallback_patterns'."
        MISSING_MARKERS+=("$outfile :: $name :: $accession :: $fallback_patterns")
        return 1
    else
        echo "  - $name ($outfile) already exists."
    fi
}

# Check for hmmfetch again before loop
if ! command -v hmmfetch &> /dev/null; then
    echo "ERROR: hmmfetch (HMMER) is required to extract profiles. Please install HMMER."
    exit 1
fi

# --- 1. Core Methanogenesis ---
fetch_hmm "mcrA.hmm" "TIGR03256" "Methyl-coM reductase alpha"
fetch_hmm "mcrB.hmm" "TIGR03258" "mcrA partner (beta)"
fetch_hmm "mcrG.hmm" "TIGR03259" "mcrA partner (gamma)"

# --- 2. The Sulfate Bypass (Mangrove Critical) ---
# Essential for capturing methylotrophic methanogenesis which competes with sulfate reducers
fetch_hmm "mtaB.hmm" "TIGR02626" "Methanol utilization" "mtaB|methanol|methanol.*methyltransferase|mta"
fetch_hmm "mttB.hmm" "TIGR02512" "Methylamine utilization" "mttB|methylamine|trimethylamine|mtt"
fetch_hmm "mtbA.hmm" "TIGR02506" "Methylcobalamin:CoM methyltransferase" "mtbA|methylcobalamin|cob|methyltransferase.*com"

# --- 3. The Copper-Switch Oxidizers ---
fetch_hmm "pmoA.hmm" "TIGR03080" "Particulate methane monooxygenase A" "pmoA|particulate methane monooxygenase"
fetch_hmm "mmoX.hmm" "TIGR01691" "Soluble methane monooxygenase" "mmoX|soluble methane monooxygenase"

# --- 4. The Competitors (Sulfate Reducers) ---
fetch_hmm "dsrA.hmm" "TIGR02064" "Dissimilatory sulfite reductase alpha" "dsrA|dissimilatory sulfite reductase"
fetch_hmm "dsrB.hmm" "TIGR02066" "Dissimilatory sulfite reductase beta" "dsrB|dissimilatory sulfite reductase"

# --- 5. Controls & Normalization ---
fetch_hmm "nifH.hmm" "TIGR01287" "Nitrogenase" "nifH|nitrogenase"
fetch_hmm "cbbL.hmm" "TIGR01168" "RuBisCO large" "cbbL|rubisco|ribulose"

echo "Success! All strategic HMM markers are ready in data/hmm/"

if [ ${#MISSING_MARKERS[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: Some marker profiles could not be extracted from TIGRFAMs_15.0_HMM.LIB."
    echo "This usually means the TIGR accession was renamed/retired or the marker is not present in this release."
    echo ""
    echo "Unresolved markers:"
    for item in "${MISSING_MARKERS[@]}"; do
        echo "  - $item"
    done
    echo ""
    echo "Next debug step (run from repo root):"
    echo "  grep -iE '<pattern>' data/hmm/TIGRFAMs_15.0_HMM.LIB.keys | head"
    echo ""
    exit 1
fi

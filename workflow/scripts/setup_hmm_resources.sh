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

# Optional: pin Pfam release for fallback extraction.
# This is used when a marker model is not present in TIGRFAMs 15.0.
PFAM_RELEASE=${PFAM_RELEASE:-"37.0"}
PFAM_A_HMM="Pfam-A.hmm"

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
    # hmmfetch -l is not available in all HMMER builds; use hmmstat instead.
    # hmmstat tbl columns: name (col2), accession (col3)
    hmmstat --tblout /dev/stdout TIGRFAMs_15.0_HMM.LIB | awk '!/^#/{print $3, $2}' > TIGRFAMs_15.0_HMM.LIB.keys
fi

# Download Pfam-A library for fallback (version pinned via PFAM_RELEASE)
# Only used if TIGRFAM extraction fails for a marker.
download_pfam() {
    if [ -f "$PFAM_A_HMM" ] || [ -f "${PFAM_A_HMM}.gz" ]; then
        return 0
    fi

    echo "Downloading Pfam-A HMM library (release ${PFAM_RELEASE}) for fallback..."
    local url="https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam${PFAM_RELEASE}/Pfam-A.hmm.gz"
    if command -v wget &> /dev/null; then
        wget -nc "$url" -O "${PFAM_A_HMM}.gz"
    else
        curl -L -o "${PFAM_A_HMM}.gz" "$url"
    fi
    gunzip -f "${PFAM_A_HMM}.gz"
}

index_library() {
    local libfile=$1
    local keysfile=$2

    if [ ! -f "${libfile}.ssi" ]; then
        echo "Indexing HMM library: ${libfile}"
        hmmfetch --index "$libfile"
    fi

    if [ ! -f "$keysfile" ]; then
        echo "Caching keys: $keysfile"
        hmmstat --tblout /dev/stdout "$libfile" | awk '!/^#/{print $3, $2}' > "$keysfile"
    fi
}

resolve_and_fetch() {
    local outfile=$1
    local libfile=$2
    local keysfile=$3
    local accession=$4
    local name=$5
    local fallback_patterns=$6

    if hmmfetch -o "$outfile" "$libfile" "$accession" 2>/dev/null; then
        return 0
    fi

    if [ -z "$fallback_patterns" ]; then
        fallback_patterns="$name"
    fi

    local resolved_acc=""
    local pat
    IFS='|' read -r -a pat_arr <<< "$fallback_patterns"
    for pat in "${pat_arr[@]}"; do
        pat=$(echo "$pat" | xargs)
        if [ -z "$pat" ]; then
            continue
        fi

        # Collect up to 5 candidate accessions so we can debug ambiguous matches.
        local candidates
        candidates=$(awk -v pat="$pat" 'BEGIN{IGNORECASE=1} $0 ~ pat {print $1}' "$keysfile" | head -n 5)

        if [ -n "$candidates" ]; then
            resolved_acc=$(echo "$candidates" | head -n 1)
            echo "  - Fallback matched accession $resolved_acc for '$name' in $libfile (pattern: $pat)"
            if [ $(echo "$candidates" | wc -l) -gt 1 ]; then
                echo "    Note: multiple candidates found for pattern '$pat' in $libfile; using first. Top candidates:"
                echo "$candidates" | sed 's/^/      - /'
            fi
            hmmfetch -o "$outfile" "$libfile" "$resolved_acc"
            return 0
        fi
    done

    return 1
}

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
        # Try TIGRFAMs first
        if [ ! -f "TIGRFAMs_15.0_HMM.LIB.keys" ]; then
            hmmstat --tblout /dev/stdout TIGRFAMs_15.0_HMM.LIB | awk '!/^#/{print $3, $2}' > TIGRFAMs_15.0_HMM.LIB.keys
        fi

        if resolve_and_fetch "$outfile" "TIGRFAMs_15.0_HMM.LIB" "TIGRFAMs_15.0_HMM.LIB.keys" "$accession" "$name" "$fallback_patterns"; then
            return 0
        fi

        echo "WARNING: Unable to extract '$name' from TIGRFAMs 15.0. Trying Pfam-A (release ${PFAM_RELEASE}) fallback..."
        download_pfam
        index_library "$PFAM_A_HMM" "${PFAM_A_HMM}.keys"

        if resolve_and_fetch "$outfile" "$PFAM_A_HMM" "${PFAM_A_HMM}.keys" "$accession" "$name" "$fallback_patterns"; then
            return 0
        fi

        echo "ERROR: Could not extract '$name' from TIGRFAMs 15.0 or Pfam-A ${PFAM_RELEASE}."
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

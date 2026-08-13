#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 URL OUTPUT EXPECTED_BYTES SEGMENTS" >&2
    exit 2
fi

url=$1
output=$2
expected_bytes=$3
segments=$4

if [[ ! "$expected_bytes" =~ ^[0-9]+$ ]] || (( expected_bytes <= 0 )); then
    echo "EXPECTED_BYTES must be a positive integer" >&2
    exit 2
fi
if [[ ! "$segments" =~ ^[0-9]+$ ]] || (( segments <= 0 )); then
    echo "SEGMENTS must be a positive integer" >&2
    exit 2
fi

output_dir=$(dirname "$output")
output_name=$(basename "$output")
parts_dir="${output_dir}/.${output_name}.parts"
assembling="${output}.assembling"
mkdir -p "$output_dir" "$parts_dir"

if [[ -f "$output" ]]; then
    existing_size=$(stat -c %s "$output")
    if (( existing_size == expected_bytes )); then
        echo "Already complete: $output ($existing_size bytes)"
        exit 0
    fi
    echo "Existing output has unexpected size: $existing_size != $expected_bytes" >&2
    exit 1
fi

chunk_bytes=$(( (expected_bytes + segments - 1) / segments ))

download_part() {
    local index=$1
    local start=$2
    local end=$3
    local expected=$(( end - start + 1 ))
    local part
    local temporary
    part=$(printf '%s/part-%03d' "$parts_dir" "$index")
    temporary="${part}.download"

    if [[ -f "$part" ]] && (( $(stat -c %s "$part") == expected )); then
        echo "part $index already complete"
        return 0
    fi

    rm -f "$temporary"
    curl \
        --fail \
        --location \
        --retry 8 \
        --retry-delay 2 \
        --retry-all-errors \
        --silent \
        --show-error \
        --range "${start}-${end}" \
        --output "$temporary" \
        "$url"

    local actual
    actual=$(stat -c %s "$temporary")
    if (( actual != expected )); then
        echo "part $index size mismatch: $actual != $expected" >&2
        return 1
    fi
    mv "$temporary" "$part"
    echo "part $index complete: bytes ${start}-${end}"
}

pids=()
index=0
start=0
while (( start < expected_bytes )); do
    end=$(( start + chunk_bytes - 1 ))
    if (( end >= expected_bytes )); then
        end=$(( expected_bytes - 1 ))
    fi
    download_part "$index" "$start" "$end" &
    pids+=("$!")
    index=$(( index + 1 ))
    start=$(( end + 1 ))
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if (( failed != 0 )); then
    echo "At least one range download failed; completed parts were preserved." >&2
    exit 1
fi

: > "$assembling"
for part in "$parts_dir"/part-*; do
    cat "$part" >> "$assembling"
done

assembled_size=$(stat -c %s "$assembling")
if (( assembled_size != expected_bytes )); then
    echo "Assembled size mismatch: $assembled_size != $expected_bytes" >&2
    exit 1
fi
mv "$assembling" "$output"
rm -f "$parts_dir"/part-*
rmdir "$parts_dir"
echo "Download complete: $output ($assembled_size bytes)"

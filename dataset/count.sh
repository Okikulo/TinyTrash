#!/bin/zsh

echo "================================"
echo "TinyTrash Dataset Photo Count"
echo "================================"

total=0

for category in metal paper plastic glass others; do
    if [ -d "$category" ]; then
        count=$(find "$category" -type f \( -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
        printf "%-10s: %d\n" "$category" "$count"
        total=$((total + count))
    else
        printf "%-10s: (not found)\n" "$category"
    fi
done

echo "--------------------------------"
printf "%-10s: %d\n" "Total" "$total"
echo "================================"

#!/bin/sh
set -eu
bundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$bundle_dir"
expected=36d550a6860c222952c501e41277a81d7cbc603bda8999580ae3b658c65e3717
output=vial-inspector-gb10
if [ -f "$output" ]; then
    printf '%s  %s\n' "$expected" "$output" | sha256sum -c -
    chmod +x "$output"
    echo "Executable already restored."
    exit 0
fi
for part in vial-inspector-gb10.part01 vial-inspector-gb10.part02; do
    if [ ! -f "$part" ]; then
        echo "Missing $part. Run git lfs pull first." >&2
        exit 1
    fi
done
temporary="$output.rebuilding.$$"
trap 'rm -f -- "$temporary"' EXIT HUP INT TERM
cat -- vial-inspector-gb10.part01 vial-inspector-gb10.part02 > "$temporary"
printf '%s  %s\n' "$expected" "$temporary" | sha256sum -c -
chmod +x "$temporary"
mv -- "$temporary" "$output"
echo "Restored $output."

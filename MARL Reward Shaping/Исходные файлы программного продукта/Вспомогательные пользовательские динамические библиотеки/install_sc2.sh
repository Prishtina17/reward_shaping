#!/usr/bin/env bash
# Install SC2 and add the custom maps (SMAC).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
smac_maps="${SCRIPT_DIR}/smac_maps"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

fetch() {
        local url="$1"
        local out="$2"
        if have_cmd wget; then
                wget -O "$out" "$url"
        elif have_cmd curl; then
                curl -L -o "$out" "$url"
        else
                echo "Ошибка: не найдено ни wget, ни curl. Установите один из них и повторите запуск." >&2
                return 127
        fi
}

cd "$HOME"
SC2PATH="${SC2PATH:-$HOME/StarCraftII}"
export SC2PATH
echo "SC2PATH is set to ${SC2PATH}"

install_root="$(dirname "${SC2PATH}")"
mkdir -p "${install_root}"
cd "${install_root}"

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        fetch http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip SC2.4.10.zip
        unzip -P iagreetotheeula SC2.4.10.zip
        rm -rf SC2.4.10.zip
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

fetch https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip SMAC_Maps.zip
unzip SMAC_Maps.zip

cp -r $smac_maps/* ./SMAC_Maps 
mv SMAC_Maps $MAP_DIR
rm -rf SMAC_Maps.zip


echo 'StarCraft II and SMAC are installed.'

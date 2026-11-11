#!/usr/bin/env bash
# Downloads every raw file the project uses into data/raw (~1.3 GB, gitignored).
# Usage: bash scripts/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/../data/raw" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data/raw"; cd "$(dirname "$0")/../data/raw"; }

FIN=https://data.gov.ro/dataset/d3caacb6-2c08-445e-94e6-8d36d00ab250/resource   # situatii financiare 2024
ONRC=https://data.gov.ro/dataset/f7374920-a656-4e34-85dd-a61c6e6e5603/resource  # firme inregistrate 08.12.2025
NOM=https://data.gov.ro/dataset/e7767b53-5749-469d-8a73-f73e1bf05ba7/resource   # older ONRC dump; only its status nomenclator is used

get() { [ -s "$2" ] && echo "skip $2" || { echo "get  $2"; curl -fsSL --retry 3 -o "$2" "$1"; }; }

# Client list
get "https://drive.google.com/uc?export=download&id=1_E2o7DrlB8AfxXNnaa2qzZV85G1yGaCA" licensee_list.csv
# Financial statements 2024 (.txt = data, .csv = column legend)
get $FIN/25098618-f6a5-4610-8c7f-c0bdb801635f/download/web_uu_an2024.txt          web_uu_an2024.txt
get $FIN/7bb0d893-2ba7-46d7-aeb0-07fb310c9a02/download/web_uu_an2024.csv          legend_web_uu_an2024.csv
get $FIN/b9f399d8-b641-4a23-9de7-a1dd4427b4b0/download/web_bl_bs_sl_an2024.csv    legend_web_bl_bs_sl_an2024.csv
get "$(curl -fsSL https://data.gov.ro/dataset/situatii_financiare_2024 | grep -oE 'https://[^"]+download/web_bl_bs_sl_an2024.txt' | head -1)" web_bl_bs_sl_an2024.txt
get $FIN/75109fe1-095a-4153-90c1-66b8e9dbbf3f/download/descriere-fisiere-web-bilant.txt descriere-fisiere-web-bilant.txt
# Trade register snapshot
get $ONRC/488a8d00-90df-4f37-b5f4-6c9111e6f1e7/download/od_firme.csv              od_firme.csv
get $ONRC/9ab6d186-6cf9-4330-b2ee-430469501f9d/download/od_stare_firma.csv        od_stare_firma.csv
get $ONRC/7001180a-9be1-4951-bc06-9a91057ce70b/download/od_caen_autorizat.csv     od_caen_autorizat.csv
get $ONRC/7b81f1a6-cd77-4b01-bf92-2795653a69e5/download/od_reprezentanti_legali.csv od_reprezentanti_legali.csv
get $NOM/c9b985d1-49b2-4a24-a325-38bc03096d60/download/5nomenclatorstarifirma.csv  nomenclator_stari_firma.csv
echo done

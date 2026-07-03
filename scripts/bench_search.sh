#!/usr/bin/env bash
# Benchmark de latencia de búsqueda contra un documento ya 'complete'.
# Uso: ./scripts/bench_search.sh <document_id> [query] [n_requests] [host]
set -euo pipefail

DOC_ID="${1:?Uso: bench_search.sh <document_id> [query] [n] [host]}"
QUERY="${2:-contrato}"
N="${3:-20}"
HOST="${4:-http://localhost:8000}"

URL="$HOST/simple/search/$DOC_ID?query=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$QUERY")&k=3"

echo "Doc: $DOC_ID | Query: \"$QUERY\" | $N requests | $HOST"
echo

# Primera request aparte: puede ser cache miss (descarga + deserialización)
t=$(curl -s -o /dev/null -w "%{time_total}" "$URL")
echo "Request 1 (posible cache miss): ${t}s"
echo

# Resto: caché caliente
times=()
for i in $(seq 2 "$N"); do
  t=$(curl -s -o /dev/null -w "%{time_total}" "$URL")
  times+=("$t")
  printf "Request %-3s %ss\n" "$i:" "$t"
done

printf '%s\n' "${times[@]}" | sort -n | awk '
  { a[NR]=$1; sum+=$1 }
  END {
    printf "\nCaché caliente (n=%d): media=%.3fs  p50=%.3fs  p95=%.3fs  min=%.3fs  max=%.3fs\n",
      NR, sum/NR, a[int(NR*0.5)+1], a[int(NR*0.95)==0?1:int(NR*0.95)], a[1], a[NR]
  }'

echo
echo "Regla de decisión: si p50 caliente > ~150ms en tus docs típicos,"
echo "vale implementar el caché de tensores en GPU; si no, no lo toques."

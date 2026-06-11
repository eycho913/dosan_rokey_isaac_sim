#!/bin/bash
# packages_2026-06-08.csv 데이터 기반 시뮬 큐 전송 스크립트
# 사용법: bash send_packages.sh            (전체 순차 전송, 1초 간격)
#          bash send_packages.sh 3          (3번 패키지만 전송)
#          bash send_packages.sh 1 5        (1~5번 범위 전송)

QUEUE=/tmp/sh5_queue.jsonl

send() {
  echo "$1" >> $QUEUE
  echo "[전송] $1"
}

PKG1='{"package_id":"PKG_20260608_001","qr_id":"QR_20260608_001","customer_name":"서도윤","target_line":"sg2_in_01"}'
PKG2='{"package_id":"PKG_20260608_002","qr_id":"QR_20260608_002","customer_name":"권하은","target_line":"sg2_in_01"}'
PKG3='{"package_id":"PKG_20260608_003","qr_id":"QR_20260608_003","customer_name":"강민준","target_line":"sg2_in_01"}'
PKG4='{"package_id":"PKG_20260608_004","qr_id":"QR_20260608_004","customer_name":"강민서","target_line":"sg2_in_01"}'
PKG5='{"package_id":"PKG_20260608_005","qr_id":"QR_20260608_005","customer_name":"전윤서","target_line":"sg2_in_01"}'
PKG6='{"package_id":"PKG_20260608_006","qr_id":"QR_20260608_006","customer_name":"신윤서","target_line":"sg2_in_01"}'
PKG7='{"package_id":"PKG_20260608_007","qr_id":"QR_20260608_007","customer_name":"한예준","target_line":"sg2_in_01"}'
PKG8='{"package_id":"PKG_20260608_008","qr_id":"QR_20260608_008","customer_name":"권주원","target_line":"sg2_in_01"}'
PKG9='{"package_id":"PKG_20260608_009","qr_id":"QR_20260608_009","customer_name":"전주원","target_line":"sg2_in_01"}'
PKG10='{"package_id":"PKG_20260608_010","qr_id":"QR_20260608_010","customer_name":"안지호","target_line":"sg2_in_01"}'
PKG11='{"package_id":"PKG_20260608_011","qr_id":"QR_20260608_011","customer_name":"임시우","target_line":"sg2_in_01"}'
PKG12='{"package_id":"PKG_20260608_012","qr_id":"QR_20260608_012","customer_name":"이지후","target_line":"sg2_in_01"}'
PKG13='{"package_id":"PKG_20260608_013","qr_id":"QR_20260608_013","customer_name":"이서현","target_line":"sg2_in_01"}'
PKG14='{"package_id":"PKG_20260608_014","qr_id":"QR_20260608_014","customer_name":"송수아","target_line":"sg2_in_01"}'
PKG15='{"package_id":"PKG_20260608_015","qr_id":"QR_20260608_015","customer_name":"박시우","target_line":"sg2_in_01"}'
PKG16='{"package_id":"PKG_20260608_016","qr_id":"QR_20260608_016","customer_name":"이지후","target_line":"sg2_in_01"}'
PKG17='{"package_id":"PKG_20260608_017","qr_id":"QR_20260608_017","customer_name":"오민서","target_line":"sg2_in_01"}'
PKG18='{"package_id":"PKG_20260608_018","qr_id":"QR_20260608_018","customer_name":"홍지후","target_line":"sg2_in_01"}'
PKG19='{"package_id":"PKG_20260608_019","qr_id":"QR_20260608_019","customer_name":"황지민","target_line":"sg2_in_01"}'
PKG20='{"package_id":"PKG_20260608_020","qr_id":"QR_20260608_020","customer_name":"권민준","target_line":"sg2_in_01"}'

PKGS=($PKG1 $PKG2 $PKG3 $PKG4 $PKG5 $PKG6 $PKG7 $PKG8 $PKG9 $PKG10
       $PKG11 $PKG12 $PKG13 $PKG14 $PKG15 $PKG16 $PKG17 $PKG18 $PKG19 $PKG20)

# 인자 파싱
if [ $# -eq 0 ]; then
  # 전체 순차 전송
  echo "=== 전체 20개 패키지 순차 전송 (간격: 1초) ==="
  for pkg in "${PKGS[@]}"; do
    send "$pkg"
    sleep 1
  done
elif [ $# -eq 1 ]; then
  # 특정 번호 1개
  idx=$1
  varname="PKG${idx}"
  send "${!varname}"
elif [ $# -eq 2 ]; then
  # 범위 전송
  echo "=== ${1}번 ~ ${2}번 패키지 전송 (간격: 1초) ==="
  for ((i=$1; i<=$2; i++)); do
    varname="PKG${i}"
    send "${!varname}"
    sleep 1
  done
fi

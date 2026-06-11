#!/usr/bin/env bash
# ============================================================
# sh5_bringup_ros2_3robot.py 테스트 트리거 스크립트
# 3개 라인(sg2_in_01~03)에 순차 패키지 투입 (총 9개, 3배치)
# ============================================================

QUEUE_FILE="/tmp/sh5_queue.jsonl"
BATCH_WAIT=10   # 배치 사이 대기 시간 (초) — 너무 짧으면 이전 상자가 처리 중에 다음 상자 투입됨

send() {
    local payload="$1"
    echo "$payload" >> "$QUEUE_FILE"
    echo "[$(date '+%H:%M:%S')] 투입: $payload"
}

echo "======================================================"
echo " SH5 3-Robot 트리거 테스트 시작 (배치 대기=${BATCH_WAIT}s)"
echo "======================================================"

# ── 배치 1 ────────────────────────────────────────────────
echo ""
echo "▶ [배치 1] 3개 라인 동시 투입"
send '{"package_id":"PKG_001","qr_id":"QR_001","customer_id":"CUST_A","target_line":"sg2_in_01"}'
send '{"package_id":"PKG_002","qr_id":"QR_002","customer_id":"CUST_B","target_line":"sg2_in_02"}'
send '{"package_id":"PKG_003","qr_id":"QR_003","customer_id":"CUST_C","target_line":"sg2_in_03"}'
echo "  → ${BATCH_WAIT}초 대기 중..."
sleep "$BATCH_WAIT"

# ── 배치 2 ────────────────────────────────────────────────
echo ""
echo "▶ [배치 2] 3개 라인 동시 투입"
send '{"package_id":"PKG_004","qr_id":"QR_004","customer_id":"CUST_A2","target_line":"sg2_in_01"}'
send '{"package_id":"PKG_005","qr_id":"QR_005","customer_id":"CUST_B2","target_line":"sg2_in_02"}'
send '{"package_id":"PKG_006","qr_id":"QR_006","customer_id":"CUST_C2","target_line":"sg2_in_03"}'
echo "  → ${BATCH_WAIT}초 대기 중..."
sleep "$BATCH_WAIT"

# ── 배치 3 ────────────────────────────────────────────────
echo ""
echo "▶ [배치 3] 3개 라인 동시 투입"
send '{"package_id":"PKG_007","qr_id":"QR_007","customer_id":"CUST_A3","target_line":"sg2_in_01"}'
send '{"package_id":"PKG_008","qr_id":"QR_008","customer_id":"CUST_B3","target_line":"sg2_in_02"}'
send '{"package_id":"PKG_009","qr_id":"QR_009","customer_id":"CUST_C3","target_line":"sg2_in_03"}'

echo ""
echo "======================================================"
echo " ✅ 전체 9개 패키지 투입 완료"
echo "======================================================"

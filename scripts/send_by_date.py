import csv
import json
import time

# CSV 데이터 (입출고일 기반 타겟 라인 설정)
csv_data = """package_id,customer_name,route_zone,qr_id
PKG_20260608_001,서도윤,2026-06-09,QR_20260608_001
PKG_20260608_002,권하은,2026-06-08,QR_20260608_002
PKG_20260608_003,강민준,2026-06-08,QR_20260608_003
PKG_20260608_004,강민서,2026-06-08,QR_20260608_004
PKG_20260608_005,전윤서,2026-06-10,QR_20260608_005
PKG_20260608_006,신윤서,2026-06-09,QR_20260608_006
PKG_20260608_007,한예준,2026-06-08,QR_20260608_007
PKG_20260608_008,권주원,2026-06-09,QR_20260608_008
PKG_20260608_009,전주원,2026-06-09,QR_20260608_009
PKG_20260608_010,안지호,2026-06-08,QR_20260608_010
PKG_20260608_011,임시우,2026-06-09,QR_20260608_011
PKG_20260608_012,이지후,2026-06-09,QR_20260608_012
PKG_20260608_013,이서현,2026-06-09,QR_20260608_013
PKG_20260608_014,송수아,2026-06-08,QR_20260608_014
PKG_20260608_015,박시우,2026-06-08,QR_20260608_015
PKG_20260608_016,이지후,2026-06-09,QR_20260608_016
PKG_20260608_017,오민서,2026-06-08,QR_20260608_017
PKG_20260608_018,홍지후,2026-06-08,QR_20260608_018
PKG_20260608_019,황지민,2026-06-08,QR_20260608_019
PKG_20260608_020,권민준,2026-06-08,QR_20260608_020"""

QUEUE_FILE = "/tmp/sh5_queue.jsonl"
BATCH_WAIT = 5  # 각 상자를 투입하는 간격(초). 로봇 처리 속도에 맞춰 조절 가능.

# 날짜에 따른 타겟 라인 매핑
ROUTE_TO_LINE = {
    "2026-06-08": "sg2_in_01", # 오늘 (0608)
    "2026-06-09": "sg2_in_02", # 내일 (0609)
    "2026-06-10": "sg2_in_03"  # 모레 (0610)
}

def send_packages():
    lines = csv_data.strip().split('\n')
    reader = csv.DictReader(lines)
    
    print("======================================================")
    print(" 📦 입출고일 기준 상자 투입 시작 (총 20개)")
    print("======================================================")

    for idx, row in enumerate(reader):
        route_date = row["route_zone"]
        target_line = ROUTE_TO_LINE.get(route_date)
        
        if not target_line:
            print(f"[경고] 알 수 없는 날짜: {route_date}. 스킵합니다.")
            continue
            
        payload = {
            "package_id": row["package_id"],
            "qr_id": row["qr_id"],
            "customer_id": row["customer_name"],
            "target_line": target_line
        }
        
        # 파일 큐에 JSON 데이터 추가
        with open(QUEUE_FILE, "a") as f:
            f.write(json.dumps(payload) + "\n")
            
        print(f"[{idx+1}/20] 투입 완료 -> {target_line} (날짜: {route_date}, 고객: {row['customer_name']})")
        time.sleep(BATCH_WAIT)
        
    print("======================================================")
    print(" ✅ 전체 20개 패키지 투입 완료")
    print("======================================================")

if __name__ == "__main__":
    send_packages()

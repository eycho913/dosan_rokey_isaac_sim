import h5py
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="HDF5 에피소드 정제 스크립트 (스텝 수 및 특정 에피소드 삭제)")
    parser.add_argument("--input", type=str, required=True, help="원본 HDF5 파일 경로")
    parser.add_argument("--output", type=str, required=True, help="필터링된 데이터가 저장될 파일 경로")
    parser.add_argument("--min_steps", type=int, default=0, help="이 숫자 미만의 스텝을 가진 에피소드는 삭제 (기본값: 0)")
    parser.add_argument("--remove_episodes", nargs='+', default=[], help="삭제할 에피소드 이름 목록 (예: demo_2 demo_5)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"파일을 찾을 수 없습니다: {args.input}")
        return

    with h5py.File(args.input, 'r') as f_in, h5py.File(args.output, 'w') as f_out:
        if 'data' not in f_in:
            print("[에러] 원본 파일에 'data' 그룹이 없습니다.")
            return
            
        data_in = f_in['data']
        data_out = f_out.create_group('data')
        
        kept = 0
        removed = 0
        
        print(f"[필터링 시작]")
        if args.min_steps > 0:
            print(f" - {args.min_steps} 스텝 미만 자동 삭제")
        if args.remove_episodes:
            print(f" - 지정 삭제 대상: {args.remove_episodes}")
        print("-" * 50)
        
        for demo_name in sorted(data_in.keys(), key=lambda x: int(x.split('_')[1])):
            demo = data_in[demo_name]
            num_samples = demo.attrs.get('num_samples', 0)
            
            # 삭제 조건 판별
            is_too_short = num_samples < args.min_steps
            is_targeted_for_removal = demo_name in args.remove_episodes
            
            if is_too_short or is_targeted_for_removal:
                reason = "수동 지정 삭제" if is_targeted_for_removal else "기준 미달"
                print(f"❌ 삭제: {demo_name} ({num_samples} steps) - {reason}")
                removed += 1
            else:
                print(f"✅ 유지: {demo_name} ({num_samples} steps)")
                # 에피소드 전체 복사
                data_in.copy(demo_name, data_out)
                
                # 에피소드 번호 빈틈없이 재정렬
                new_demo_name = f"demo_{kept}"
                if new_demo_name != demo_name:
                    data_out.move(demo_name, new_demo_name)
                    
                kept += 1
                
        print("-" * 50)
        print(f"🎉 정리 완료! (유지된 에피소드: {kept}개, 삭제된 에피소드: {removed}개)")
        print(f"📁 새 파일 저장 위치: {args.output}")

if __name__ == "__main__":
    main()

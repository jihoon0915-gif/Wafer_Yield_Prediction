from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# 확정된 공정 순서 (조인 시 이 순서로 병합하여, 중간에 행 수가 깨지면 원인 파일을 바로 특정할 수 있게 함)
PROCESS_ORDER = [
    "Photo_softbake.csv",
    "Photo_lithograpy.csv",  # 원본 파일명 오타(lithograpy) 그대로 사용
    "Etching.csv",
    "Ion_Implantation.csv",
    "Oxidation.csv",
    "Inspect.csv",
]

# 6개 파일에 공통으로 존재하는 조인/식별 컬럼
COMMON_KEY_COLS = ["No_Die", "Lot_Num", "Wafer_Num", "Datetime"]

EXPECTED_ROW_COUNT = 15_390
EXPECTED_GROUP_COUNT = 1_704
WAFER_MAP_GRID_SIZE = 26
WAFER_MAP_REAL_DIE_COUNT = 533

# Error_message 8개 클래스 (WM-811K 체계 중 이 데이터셋에 존재하는 7개 결함 + none)
ERROR_CLASSES = [
    "none",
    "Center",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
]

# pptx 변수정의서(260320_...변수정의서(B반_반도체).pptx) 기준 물리 스펙
OXIDATION_THICKNESS_MIN_NM = 700.0
LITHOGRAPHY_LINE_CD_RANGE_NM = (25.0, 55.0)

# --- 0/음수 센티널 값 스캔 대상 -------------------------------------------
# 각 CSV의 컬럼 중 조인키/ID/범주형(Chamber 번호, process 단계명 등)을 제외한
# "수치형 공정변수"만 모은 목록. <=0 값 개수를 스캔해 보고하는 넓은 범위.
SENTINEL_SCAN_COLUMNS = [
    # Photo_softbake.csv
    "resist_target", "N2_HMDS", "pressure_HMDS", "temp_HMDS", "temp_HMDS_bake",
    "time_HMDS_bake", "spin1", "spin2", "spin3", "photoresist_bake",
    "temp_softbake", "time_softbake",
    # Photo_lithograpy.csv
    "Line_CD", "Wavelength", "Resolution", "Energy_Exposure",
    # Etching.csv
    "Thin F1", "Thin F2", "Thin F3", "Thin F4", "Temp_Etching", "Source_Power",
    "Selectivity",
    # Ion_Implantation.csv
    "Flux60s", "Flux90s", "Flux160s", "Flux480s", "Flux840s", "input_Energy",
    "Temp_implantation", "Furance_Temp", "RTA_Temp",
    # Oxidation.csv
    "Temp_OXid", "ppm", "Pressure", "Oxid_time", "thickness",
]

# 위 스캔 대상 중, "물리적으로 0/음수가 나올 수 없는" 온도·압력·시간·파워·두께·
# 파장·에너지·flux 계열만 추린 잠정(provisional) 변환 후보. resist_target(1 근방
# 비율 지표)·Resolution·Selectivity(비율형 출력 지표)는 0/음수가 곧 결측이라고
# 단정할 근거가 약해 제외 — pptx 정의서에 "0/음수=결측" 규칙이 명시돼 있지 않으므로
# 최종 확정 전 사용자 확인이 필요하다.
SENTINEL_CONVERT_CANDIDATE_COLUMNS = [
    "N2_HMDS", "pressure_HMDS", "temp_HMDS", "temp_HMDS_bake", "time_HMDS_bake",
    "spin1", "spin2", "spin3", "photoresist_bake", "temp_softbake", "time_softbake",
    "Wavelength", "Energy_Exposure",
    "Thin F1", "Thin F2", "Thin F3", "Thin F4", "Temp_Etching", "Source_Power",
    "Flux60s", "Flux90s", "Flux160s", "Flux480s", "Flux840s", "input_Energy",
    "Temp_implantation", "Furance_Temp", "RTA_Temp",
    "Temp_OXid", "ppm", "Pressure", "Oxid_time", "thickness",
]

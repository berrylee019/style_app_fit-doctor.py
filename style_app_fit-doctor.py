import streamlit as st
import google.generativeai as genai
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from PIL import Image

# 1. 초기 설정 (Secrets 및 페이지 설정)
try:
    genai.configure(api_key=st.secrets["MY_API_KEY"])
except Exception as e:
    st.error(f"⚠️ API 키 설정 오류: Streamlit Secrets를 확인해주세요. ({e})")

st.set_page_config(page_title="AI 바디 밸런스 코치", page_icon="🏋️", layout="wide")

# --- [중요] MediaPipe Pose 설정 섹션 (서버 환경 최적화) ---
# 가장 범용적인 임포트 방식을 사용하여 경로 에러를 원천 차단합니다.
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 관절 분석기 초기화 (메모리 효율을 위해 한 번만 실행)
@st.cache_resource
def load_pose_model():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose = load_pose_model()

# --- [수익화 링크 설정] ---
MY_REVENUE_LINK = "https://link.inpock.co.kr/shopping1" # 형님의 수익 링크

# --- [함수] 영상 관절 데이터 추출 ---
def analyze_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ratios = []
    
    # 영상 전체 프레임 수 확인
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return 1.0
        
    # 분석할 프레임 지점 선정 (영상 길이에 따라 유연하게 추출)
    sample_points = [frame_count//4, frame_count//2, 3*frame_count//4]
    
    for i in sample_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = cap.read()
        if not success: continue
        
        # 이미지 전처리 및 관절 추출
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # 어깨 및 골반 좌표 (안전하게 인덱스로 접근)
                l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                
                # 너비 계산 (유클리드 거리)
                sh_width = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
                hip_width = np.sqrt((l_hip.x - r_hip.x)**2 + (l_hip.y - r_hip.y)**2)
                
                if hip_width > 0:
                    ratios.append(sh_width / hip_width)
            except Exception:
                continue
            
    cap.release()
    return np.mean(ratios) if ratios else 1.0

# --- UI 레이아웃 ---
st.title("🏋️ AI 바디 밸런스 코치")
st.markdown("##### 영상 10초로 분석하는 내 몸의 황금 비율과 맞춤 운동법")
st.divider()

col_guide, col_upload = st.columns([1.3, 1])

with col_guide:
    st.markdown("### 📽️ 바디 스캔 가이드")
    st.video("https://www.youtube.com/watch?v=1vE5QSvW_Vg") 
    st.info("💡 **팁:** 전신이 다 나오도록 촬영하고, 정면이 잘 보일 때 분석이 가장 정확합니다!")

with col_upload:
    st.markdown("### 🎬 바디 스캔 시작")
    uploaded_file = st.file_uploader("분석할 영상을 업로드하세요 (MP4, MOV)", type=["mp4", "mov"])
    
    if uploaded_file:
        if st.button("🚀 AI 체형 분석 및 운동 처방 시작", use_container_width=True, type="primary"):
            with st.spinner("AI가 형님의 골격 데이터를 정밀 분석 중입니다..."):
                # 1. 임시 파일 저장 (안전한 처리)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                
                try:
                    # 2. MediaPipe 수치 데이터 추출
                    body_ratio = analyze_pose_from_video(video_path)
                    
                    # 3. Gemini 전문가 코칭 생성
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    ratio_status = "어깨가 골반보다 넓은 역삼각형" if body_ratio > 1.2 else "상하체 균형형" if body_ratio > 0.9 else "하체에 비해 상체가 왜소한 체형"
                    
                    prompt = f"""
                    사용자의 체형 수치 분석 결과: 어깨 대비 골반 비율 {body_ratio:.2f} (상태: {ratio_status}).
                    위 데이터를 바탕으로:
                    1. 이 체형이 가진 미적/기능적 장점을 칭찬해줘.
                    2. 현재 비율에서 '황금 비율'로 가기 위해 보충해야 할 운동 루틴 3가지를 아주 구체적으로 설명해줘.
                    3. 운동 시 다치지 않게 주의할 점과 도움이 되는 헬스 기구를 추천해줘.
                    
                    마지막엔 반드시 '# 추천 기구: [기구이름1, 기구이름2]' 형식으로 끝내줘.
                    형님에게 조언하는 전문 트레이너처럼 친근하고 든든하게 말해줘.
                    """
                    
                    with open(video_path, 'rb') as f:
                        video_data = f.read()
                    
                    response = model.generate_content([
                        prompt, 
                        {"mime_type": "video/mp4", "data": video_data}
                    ])
                    
                    st.session_state.body_analysis = response.text
                    st.session_state.body_stage = 'analyzed'
                    
                finally:
                    # 임시 파일 삭제 (서버 용량 관리)
                    if os.path.exists(video_path):
                        os.remove(video_path)
                
                st.rerun()

# --- 결과 및 수익화 출력 ---
if 'body_stage' in st.session_state and st.session_state.body_stage == 'analyzed':
    st.divider()
    st.subheader("📊 AI 체형 분석 및 맞춤 코칭 리포트")
    # 1. 줄바꿈 변환을 f-string 밖에서 미리 수행합니다.
    formatted_analysis = st.session_state.body_analysis.replace('\n', '<br>')
    
    # 2. 변환된 변수를 f-string 안에 넣습니다.
    # 1. 줄바꿈 변환을 f-string 밖에서 미리 수행합니다.
    formatted_analysis = st.session_state.body_analysis.replace('\n', '<br>')
    
    # 2. 변환된 변수를 f-string 안에 넣습니다.
    # 1. 줄바꿈 변환을 f-string 밖에서 미리 수행합니다.
    formatted_analysis = st.session_state.body_analysis.replace('\n', '<br>')
    
    # 2. 변환된 변수를 f-string 안에 넣습니다.
    # 1. 줄바꿈 변환을 f-string 밖에서 미리 수행합니다.
    formatted_analysis = st.session_state.body_analysis.replace('\n', '<br>')
    
    # 2. 변환된 변수를 f-string 안에 넣습니다.
    # 1. 줄바꿈 변환을 f-string 밖에서 미리 수행합니다.
    formatted_analysis = st.session_state.body_analysis.replace('\n', '<br>')
    
    # 2. 변환된 변수를 f-string 안에 넣습니다.
    st.markdown(f"""
        <div style='background-color:#F8FAFC; padding:25px; border-radius:15px; border:1px solid #E2E8F0; line-height:1.7; color:#1E293B;'>
            {formatted_analysis}
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    if st.button("✨ 운동에 필요한 기구 및 보충제 확인하기 (형님 전용 혜택)", use_container_width=True):
        st.session_state.body_stage = 'shopping'
        st.rerun()

if 'body_stage' in st.session_state and st.session_state.body_stage == 'shopping':
    st.subheader("🛒 형님을 위한 맞춤 운동 서포트 아이템")
    st.info("💡 분석 리포트에서 추천된 기구들을 아래 링크에서 최저가로 확인하세요!")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image("https://via.placeholder.com/300?text=Fitness+Gear")
        st.link_button("🔥 하체 폭발 운동 밴드", MY_REVENUE_LINK, use_container_width=True)
    with c2:
        st.image("https://via.placeholder.com/300?text=Protein+Protein")
        st.link_button("🥛 근육 생성 단백질 쉐이크", MY_REVENUE_LINK, use_container_width=True)
    with c3:
        st.image("https://via.placeholder.com/300?text=Recovery+Supplements")
        st.link_button("💪 근력 증대 필수 영양제", MY_REVENUE_LINK, use_container_width=True)
    
    st.success(f"형님, 이 루틴만 꾸준히 하시면 몸의 변화가 확실히 느껴지실 겁니다! {MY_REVENUE_LINK}에서 필요한 장비를 챙겨보세요.")

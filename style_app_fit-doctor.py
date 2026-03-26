import streamlit as st
import google.generativeai as genai
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from PIL import Image

# 1. 초기 설정 (Secrets 및 페이지 설정)
try:
    genai.configure(api_key=st.secrets["MY_API_KEY"])
except:
    st.error("⚠️ Streamlit Secrets에 'MY_API_KEY'를 설정해주세요.")

st.set_page_config(page_title="AI 바디 밸런스 코치", page_icon="🏋️", layout="wide")

# --- [중요] MediaPipe Pose 설정 섹션 (철벽 방어형) ---
# 서버 환경에 따라 경로가 달라지는 문제를 해결합니다.
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    import mediapipe as mp
    
    # 내부 경로를 직접 import 하지 않고, mp 객체를 통해 안전하게 접근합니다.
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

# 관절 분석기 초기화
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- [수익화 링크 설정] ---
MY_REVENUE_LINK = "https://link.inpock.co.kr/shopping1" # 형님의 인포크링크

# --- [함수] 영상 관절 데이터 추출 ---
def analyze_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ratios = []
    
    # 영상 전체 프레임 수 확인
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        return 1.0
        
    # 영상의 주요 지점(25%, 50%, 75%) 프레임 분석
    for i in [frame_count//4, frame_count//2, 3*frame_count//4]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = cap.read()
        if not success: continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # 어깨(11, 12) 및 골반(23, 24) 좌표 추출
            l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            
            # 너비 계산 (단순 x좌표 차이보다 정확한 유클리드 거리 사용)
            sh_width = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
            hip_width = np.sqrt((l_hip.x - r_hip.x)**2 + (l_hip.y - r_hip.y)**2)
            
            if hip_width > 0:
                ratios.append(sh_width / hip_width)
            
    cap.release()
    return np.mean(ratios) if ratios else 1.0

# --- UI 레이아웃 ---
st.title("🏋️ AI 바디 밸런스 코치")
st.markdown("##### 영상 10초로 분석하는 내 몸의 황금 비율과 맞춤 운동법")
st.divider()

col_guide, col_upload = st.columns([1.3, 1])

with col_guide:
    st.markdown("### 📽️ 바디 스캔 가이드")
    # 형님의 유튜브 가이드 영상
    st.video("https://www.youtube.com/watch?v=1vE5QSvW_Vg") 
    st.info("💡 **팁:** 전신이 다 나오도록 촬영하고, 정면/측면이 잘 보이게 서주시면 분석이 더 정확해집니다!")

with col_upload:
    st.markdown("### 🎬 바디 스캔 시작")
    uploaded_file = st.file_uploader("분석할 영상을 업로드하세요 (MP4, MOV)", type=["mp4", "mov"])
    
    if uploaded_file:
        if st.button("🚀 AI 체형 분석 및 운동 처방 시작", use_container_width=True, type="primary"):
            with st.spinner("AI가 형님의 골격 데이터를 정밀 분석 중입니다..."):
                # 임시 파일 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                
                # 1. MediaPipe 분석 (수치 데이터)
                body_ratio = analyze_pose_from_video(video_path)
                
                # 2. Gemini 1.5 Flash를 활용한 전문가 코칭 생성
                model = genai.GenerativeModel('gemini-1.5-flash')
                ratio_status = "어깨가 골반보다 넓은 역삼각형" if body_ratio > 1.2 else "상하체 균형형" if body_ratio > 0.9 else "하체에 비해 상체가 왜소한 체형"
                
                prompt = f"""
                분석 결과: 어깨 대 골반 비율 {body_ratio:.2f} ({ratio_status}).
                1. 이 체형의 시각적 특징과 장점을 전문 트레이너답게 설명해줘.
                2. 하체 근육량을 늘려 밸런스를 맞추기 위한 3단계 운동 루틴을 제안해줘.
                3. 운동 시 주의사항과 권장하는 헬스 기구를 알려줘.
                마지막에는 반드시 '# 추천 기구: [아이템1, 아이템2]' 형식으로 끝내줘.
                형님(User)에게 친근하면서도 전문적인 톤으로 작성해줘.
                """
                
                # 영상 파일과 함께 프롬프트 전달
                with open(video_path, 'rb') as f:
                    video_data = f.read()
                    
                response = model.generate_content([
                    prompt, 
                    {"mime_type": "video/mp4", "data": video_data}
                ])
                
                st.session_state.body_analysis = response.text
                st.session_state.body_stage = 'analyzed'
                st.rerun()

# --- 결과 및 수익화 출력 ---
if 'body_stage' in st.session_state and st.session_state.body_stage == 'analyzed':
    st.divider()
    st.subheader("📊 AI 체형 분석 및 맞춤 코칭 리포트")
    st.markdown(f"<div style='background-color:#F8FAFC; padding:25px; border-radius:15px; border:1px solid #E2E8F0; line-height:1.6;'>{st.session_state.body_analysis}</div>", unsafe_allow_html=True)
    
    st.write("")
    if st.button("✨ 운동에 필요한 기구 및 보충제 확인하기 (형님 전용 혜택)", use_container_width=True):
        st.session_state.body_stage = 'shopping'
        st.rerun()

if 'body_stage' in st.session_state and st.session_state.body_stage == 'shopping':
    st.subheader("🛒 형님을 위한 맞춤 운동 서포트 아이템")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image("https://via.placeholder.com/300?text=Home+Training+Gear")
        st.link_button("🔥 하체 폭발 운동 밴드", MY_REVENUE_LINK, use_container_width=True)
    with c2:
        st.image("https://via.placeholder.com/300?text=Protein+Shake")
        st.link_button("🥛 근육 생성 단백질 쉐이크", MY_REVENUE_LINK, use_container_width=True)
    with c3:
        st.image("https://via.placeholder.com/300?text=Mass+Gainer")
        st.link_button("💪 체중&근력 증대 보충제", MY_REVENUE_LINK, use_container_width=True)
    
    st.success(f"형님, 이 운동 루틴과 기구만 있으면 하체 빈약 탈출은 시간문제입니다! 모든 구매 혜택은 {MY_REVENUE_LINK}에서 확인하세요.")

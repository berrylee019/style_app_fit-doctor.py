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

# --- [중요] MediaPipe Pose 설정 섹션 (서버 환경 철벽 방어) ---
@st.cache_resource
def load_pose_engine():
# --- [중요] MediaPipe Pose 설정 섹션 (서버 환경 철벽 방어 - 최종판) ---
import mediapipe as mp

@st.cache_resource
def load_pose_engine():
    # mp.solutions를 거치지 않고 직접 내부 python 경로에서 pose와 drawing_utils를 가져옵니다.
    try:
        from mediapipe.python.solutions import pose as mp_p
        from mediapipe.python.solutions import drawing_utils as mp_d
        return mp_p, mp_d
    except ImportError:
        # 위 방식이 실패할 경우를 대비한 표준 경로 백업
        import mediapipe.solutions.pose as mp_p
        import mediapipe.solutions.drawing_utils as mp_d
        return mp_p, mp_d

# 엔진과 드로잉 유틸리티를 한 번에 가져옵니다.
mp_pose, mp_drawing = load_pose_engine()

# 관절 분석기 인스턴스 생성
@st.cache_resource
def get_pose_detector():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose_detector = get_pose_detector()

# --- [수익화 링크 설정] ---
MY_REVENUE_LINK = "https://link.inpock.co.kr/shopping1"

# --- [함수] 영상 관절 데이터 추출 ---
def analyze_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ratios = []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return 1.0
        
    # 정밀도 향상을 위해 샘플링 지점을 5곳으로 확대
    sample_points = [frame_count//6, frame_count//3, frame_count//2, 2*frame_count//3, 5*frame_count//6]
    
    for i in sample_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = cap.read()
        if not success: continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                # 랜드마크 인덱스를 명확히 지정하여 오류 방지
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
    st.info("💡 **팁:** 전신이 다 나오도록 촬영하고, 정면을 응시할 때 가장 정확합니다!")

with col_upload:
    st.markdown("### 🎬 바디 스캔 시작")
    uploaded_file = st.file_uploader("분석할 영상을 업로드하세요 (MP4, MOV)", type=["mp4", "mov"])
    
    if uploaded_file:
        if st.button("🚀 AI 체형 분석 및 운동 처방 시작", use_container_width=True, type="primary"):
            with st.spinner("AI가 형님의 골격 데이터를 정밀 분석 중입니다..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                
                try:
                    # 1. 수치 데이터 추출
                    body_ratio = analyze_pose_from_video(video_path)
                    
                    # 2. Gemini 코칭 생성
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    ratio_status = "어깨가 넓은 역삼각형" if body_ratio > 1.2 else "상하체 균형형" if body_ratio > 0.9 else "하체가 발달한 체형"
                    
                    prompt = f"""
                    체형 수치 분석 결과: 어깨 대 골반 비율 {body_ratio:.2f} ({ratio_status}).
                    전문 스포츠 트레이너로서 다음을 작성해줘:
                    1. 이 체형의 장점과 특징 (형님이라 부르며 친근하게)
                    2. 밸런스를 완성할 맞춤 운동 루틴 3단계
                    3. 권장 헬스 기구 및 주의사항
                    마지막엔 '# 추천 기구: [기구1, 기구2]' 문구를 포함해줘.
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
                    if os.path.exists(video_path):
                        os.remove(video_path)
                
                st.rerun()

# --- 결과 및 수익화 출력 ---
if 'body_stage' in st.session_state and st.session_state.body_stage == 'analyzed':
    st.divider()
    st.subheader("📊 AI 체형 분석 및 맞춤 코칭 리포트")
    
    # 안전한 문자열 변환 처리
    formatted_analysis = st.session_state.body_analysis.replace('\n', '<br>')
    
    st.markdown(f"""
        <div style='background-color:#F8FAFC; padding:25px; border-radius:15px; border:1px solid #E2E8F0; line-height:1.7; color:#1E293B;'>
            {formatted_analysis}
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    if st.button("✨ 추천 기구 및 보충제 혜택 확인하기", use_container_width=True):
        st.session_state.body_stage = 'shopping'
        st.rerun()

if 'body_stage' in st.session_state and st.session_state.body_stage == 'shopping':
    st.subheader("🛒 형님을 위한 맞춤 운동 아이템")
    
    c1, c2, c3 = st.columns(3)
    items = [
        ("https://via.placeholder.com/300?text=Fitness+Band", "🔥 하체 폭발 운동 밴드"),
        ("https://via.placeholder.com/300?text=Protein+Shake", "🥛 근육 생성 단백질 쉐이크"),
        ("https://via.placeholder.com/300?text=Supplements", "💪 근력 증대 영양제")
    ]
    
    for col, (img, name) in zip([c1, c2, c3], items):
        with col:
            st.image(img)
            st.link_button(name, MY_REVENUE_LINK, use_container_width=True)
    
    st.success(f"형님, 위 아이템들과 함께라면 득근은 시간문제입니다! 상세 혜택: {MY_REVENUE_LINK}")

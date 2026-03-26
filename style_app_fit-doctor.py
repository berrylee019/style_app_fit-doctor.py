import streamlit as st
import google.generativeai as genai
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from PIL import Image
import mediapipe.python.solutions.pose as mp_pose

# 1. 초기 설정
MY_REVENUE_LINK = "https://link.inpock.co.kr/shopping1" # 형님의 수익 링크
try:
    genai.configure(api_key=st.secrets["MY_API_KEY"])
except:
    st.error("⚠️ Streamlit Secrets에 'MY_API_KEY'를 설정해주세요.")

st.set_page_config(page_title="AI 바디 밸런스 코치", page_icon="🏋️", layout="wide")

# MediaPipe Pose 설정
# 솔루션 초기화
try:
    mp_pose = mp.solutions.pose
except AttributeError:
    
mp_drawing = mp.solutions.drawing_utils # 나중에 관절 그릴 때 필요합니다.

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- [함수] 영상 관절 데이터 추출 ---
def analyze_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ratios = []
    
    # 영상의 중간 프레임 약 3개 정도만 분석 (속도와 정확도 위함)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in [frame_count//4, frame_count//2, 3*frame_count//4]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = cap.read()
        if not success: continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # 주요 좌표 추출
            l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            
            # 어깨 너비 vs 골반 너비 비율 계산
            sh_width = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
            hip_width = np.sqrt((l_hip.x - r_hip.x)**2 + (l_hip.y - r_hip.y)**2)
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
    st.video("https://www.youtube.com/watch?v=1vE5QSvW_Vg") # 형님의 가이드 영상 그대로 활용
    st.info("💡 **팁:** 전신이 다 나오도록 촬영하고, 천천히 한 바퀴 돌면 분석이 더 정확해집니다!")

with col_upload:
    st.markdown("### 🎬 바디 스캔 시작")
    uploaded_file = st.file_uploader("분석할 영상을 업로드하세요", type=["mp4", "mov"])
    
    if uploaded_file:
        if st.button("🚀 AI 체형 분석 및 운동 처방 시작", use_container_width=True, type="primary"):
            with st.spinner("AI가 형님의 골격 데이터를 추출하고 있습니다..."):
                # 1. 임시 파일 저장 및 MediaPipe 분석
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                
                body_ratio = analyze_pose_from_video(tfile.name)
                
                # 2. Gemini에게 수치 데이터와 함께 전문가 조언 요청
                model = genai.GenerativeModel('gemini-1.5-flash')
                ratio_status = "어깨가 골반보다 넓은 역삼각형" if body_ratio > 1.2 else "상하체 균형형" if body_ratio > 0.9 else "하체에 비해 상체가 왜소한 체형"
                
                prompt = f"""
                분석된 사용자의 어깨 대 골반 비율은 {body_ratio:.2f}로, '{ratio_status}'에 해당합니다. 
                이 체형의 장점과 보완해야 할 점(특히 하체가 빈약할 경우)을 전문 트레이너처럼 설명해주고, 
                이를 개선하기 위한 3단계 운동 루틴을 아주 구체적인 방법과 함께 알려줘.
                운동기구가 필요하다면 어떤 게 좋은지도 언급해줘.
                마지막에 '# 추천 기구: [아이템1, 아이템2]' 형식으로 마무리해줘.
                """
                
                response = model.generate_content([prompt, {"mime_type": "video/mp4", "data": open(tfile.name, 'rb').read()}])
                
                st.session_state.body_analysis = response.text
                st.session_state.body_stage = 'analyzed'
                st.rerun()

# --- 결과 출력 ---
if 'body_stage' in st.session_state and st.session_state.body_stage == 'analyzed':
    st.divider()
    st.subheader("📊 AI 체형 분석 및 맞춤 코칭 리포트")
    st.markdown(f"<div style='background-color:#F0F7FF; padding:25px; border-radius:15px; border:1px solid #BFDBFE;'>{st.session_state.body_analysis}</div>", unsafe_allow_html=True)
    
    st.write("")
    if st.button("✨ 운동에 필요한 기구 및 보충제 확인하기", use_container_width=True):
        st.session_state.body_stage = 'shopping'
        st.rerun()

if 'body_stage' in st.session_state and st.session_state.body_stage == 'shopping':
    # 리포트 유지
    st.subheader("📊 AI 체형 분석 및 맞춤 코칭 리포트")
    st.info(st.session_state.body_analysis)
    
    st.divider()
    st.subheader("🛒 형님을 위한 맞춤 운동 서포트 아이템")
    # 인포크링크로 연결되는 버튼들 (수익화)
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

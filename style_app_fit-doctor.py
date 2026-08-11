import streamlit as st
import google.generativeai as genai
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import re
from PIL import Image

# 1. 초기 설정 (Secrets 및 페이지 설정)
try:
    genai.configure(api_key=st.secrets["MY_API_KEY"])
except Exception as e:
    st.error(f"⚠️ API 키 설정 오류: Streamlit Secrets를 확인해주세요. ({e})")

st.set_page_config(page_title="AI 핏 닥터 프로 & 골프 코치", page_icon="🏌️‍♂️", layout="wide")

# --- [MediaPipe Pose 설정 섹션 (서버 환경 방어)] ---
@st.cache_resource
def load_pose_engine():
    import mediapipe as mp
    try:
        mp_p = mp.solutions.pose
        mp_d = mp.solutions.drawing_utils
        return mp_p, mp_d
    except AttributeError:
        import mediapipe as mp
        mp_p = mp.solutions.pose
        from mediapipe.python.solutions import drawing_utils as mp_d
        return mp_p, mp_d

mp_pose, mp_drawing = load_pose_engine()

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

# --- [텍스트 정제 함수: 마크다운 기호 및 구분선 제거] ---
def clean_report_text(text: str) -> str:
    # 1. '---' 또는 '***' 구분을 위한 굵은 선 제거
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # 2. '###', '##', '#' 등 마크다운 헤더 기호 제거
    text = re.sub(r'#{1,6}\s*', '', text)
    # 3. '**' 굵은 글씨 특수문자 제거
    text = text.replace('**', '')
    # 4. 연속된 줄바꿈 깔끔하게 정리 후 HTML 줄바꿈으로 변환
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text.replace('\n', '<br>')

# --- [함수 1] 바디 밸런스 관절 데이터 추출 ---
def analyze_pose_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ratios = []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return 1.0
        
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
                l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                
                sh_width = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
                hip_width = np.sqrt((l_hip.x - r_hip.x)**2 + (l_hip.y - r_hip.y)**2)
                
                if hip_width > 0:
                    ratios.append(sh_width / hip_width)
            except Exception:
                continue
            
    cap.release()
    return np.mean(ratios) if ratios else 1.0

# --- [함수 2] 골프 스윙 자세 관절 데이터 추출 ---
def analyze_golf_swing_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return {"shoulder_tilt": 0.0, "hip_tilt": 0.0, "spine_angle": 0.0}
    
    sample_points = [frame_count//5, frame_count//2, 4*frame_count//5]
    shoulder_tilts, hip_tilts, spine_angles = [], [], []

    for i in sample_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = cap.read()
        if not success: continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_detector.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                
                sh_angle = np.degrees(np.arctan2(r_sh.y - l_sh.y, r_sh.x - l_sh.x))
                shoulder_tilts.append(abs(sh_angle))

                hip_angle = np.degrees(np.arctan2(r_hip.y - l_hip.y, r_hip.x - l_hip.x))
                hip_tilts.append(abs(hip_angle))

                mid_sh_x, mid_sh_y = (l_sh.x + r_sh.x)/2, (l_sh.y + r_sh.y)/2
                mid_hip_x, mid_hip_y = (l_hip.x + r_hip.x)/2, (l_hip.y + r_hip.y)/2
                spine_deg = np.degrees(np.arctan2(mid_sh_y - mid_hip_y, mid_sh_x - mid_hip_x))
                spine_angles.append(abs(90 - abs(spine_deg)))
            except Exception:
                continue

    cap.release()
    return {
        "shoulder_tilt": float(np.mean(shoulder_tilts)) if shoulder_tilts else 0.0,
        "hip_tilt": float(np.mean(hip_tilts)) if hip_tilts else 0.0,
        "spine_angle": float(np.mean(spine_angles)) if spine_angles else 0.0
    }


# --- UI 메인 타이틀 ---
st.title("🏋️ AI 핏 닥터 프로 & ⛳ 골프 스윙 코치")
st.markdown("##### 영상으로 분석하는 체형 황금 비율 & 드라이버 스윙 메커니즘 정밀 코칭")
st.divider()

tab1, tab2 = st.tabs(["🏋️ AI 바디 밸런스 코치", "⛳ 골프 스윙 / 드라이버 자세 코칭"])

# ==========================================
# [TAB 1] AI 바디 밸런스 코치
# ==========================================
with tab1:
    col_guide, col_upload = st.columns([1.3, 1])

    with col_guide:
        st.markdown("### 📽️ 바디 스캔 가이드")
        st.video("https://www.youtube.com/watch?v=1vE5QSvW_Vg") 
        st.info("💡 **팁:** 전신이 다 나오도록 촬영하고, 정면을 응시할 때 가장 정확합니다!")

    with col_upload:
        st.markdown("### 🎬 바디 스캔 시작")
        uploaded_file = st.file_uploader("분석할 영상을 업로드하세요 (MP4, MOV)", type=["mp4", "mov"], key="body_uploader")
        
        if uploaded_file:
            if st.button("🚀 AI 체형 분석 및 운동 처방 시작", use_container_width=True, type="primary", key="body_btn"):
                with st.spinner("AI가 형님의 골격 데이터를 정밀 분석 중입니다..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(uploaded_file.read())
                        video_path = tfile.name
                    
                    try:
                        body_ratio = analyze_pose_from_video(video_path)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        ratio_status = "어깨가 넓은 역삼각형" if body_ratio > 1.2 else "상하체 균형형" if body_ratio > 0.9 else "하체가 발달한 체형"
                        
                        prompt = f"""
                        체형 수치 분석 결과: 어깨 대 골반 비율 {body_ratio:.2f} ({ratio_status}).
                        전문 스포츠 트레이너로서 다음 항목에 맞춰 리포트를 작성해줘.
                        [작성 규칙]
                        - 절대로 마크다운 구분선(---), 특수문자(###, **)를 사용하지 마시오.
                        - 단락 제목은 단순히 숫자와 번호로만 구분하시오.

                        1. 체형 장점 및 특징 (형님이라 부르며 친근하게)
                        2. 밸런스 완성 맞춤 운동 루틴 3단계
                        3. 권장 헬스 기구 및 주의사항
                        """
                        
                        with open(video_path, 'rb') as f:
                            video_data = f.read()
                        
                        response = model.generate_content([
                            prompt, 
                            {"mime_type": "video/mp4", "data": video_data}
                        ])
                        
                        st.session_state.body_analysis = clean_report_text(response.text)
                        st.session_state.body_stage = 'analyzed'
                        
                    finally:
                        if os.path.exists(video_path):
                            os.remove(video_path)
                    
                    st.rerun()

    # 바디 밸런스 결과 출력
    if 'body_stage' in st.session_state and st.session_state.body_stage == 'analyzed':
        st.divider()
        st.subheader("📊 AI 체형 분석 및 맞춤 코칭 리포트")
        
        st.markdown(f"""
            <div style='background-color:#F8FAFC; padding:25px; border-radius:15px; border:1px solid #E2E8F0; line-height:1.8; color:#1E293B; font-size:15px;'>
                {st.session_state.body_analysis}
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        if st.button("✨ 추천 기구 및 보충제 혜택 확인하기", use_container_width=True, key="body_shop_btn"):
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


# ==========================================
# [TAB 2] ⛳ 골프 스윙 / 드라이버 자세 코칭
# ==========================================
with tab2:
    col_golf_guide, col_golf_upload = st.columns([1.3, 1])

    with col_golf_guide:
        st.markdown("### 📽️ 골프 스윙 촬영 가이드")
        st.video("https://www.youtube.com/watch?v=1vE5QSvW_Vg") 
        st.info("💡 **팁:** 측면(Down the line) 또는 정면(Face-on)에서 골프클럽과 머리 끝부터 발끝까지 전신이 나오도록 찍어주세요!")

    with col_golf_upload:
        st.markdown("### ⛳ 스윙 분석 시작")
        golf_file = st.file_uploader("스윙 영상 업로드 (드라이버/아이언, MP4, MOV)", type=["mp4", "mov"], key="golf_uploader")
        
        if golf_file:
            if st.button("🚀 AI 골프 스윙 정밀 진단 시작", use_container_width=True, type="primary", key="golf_btn"):
                with st.spinner("AI 프로 골프 코치가 형님의 스윙 궤적과 메커니즘을 분석 중입니다..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(golf_file.read())
                        golf_video_path = tfile.name
                    
                    try:
                        golf_metrics = analyze_golf_swing_from_video(golf_video_path)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        
                        golf_prompt = f"""
                        골프 스윙 관절 바이오매카닉 측정 결과:
                        - 어깨 회전 기울기(Tilt): 약 {golf_metrics['shoulder_tilt']:.1f}°
                        - 골반 기울기(Hip Tilt): 약 {golf_metrics['hip_tilt']:.1f}°
                        - 척추 유지 각도(Spine Angle Deviation): 약 {golf_metrics['spine_angle']:.1f}°

                        당신은 PGA 투어 출신의 전문 골프 스윙 분석 프로입니다. 형님이라 부르며 친근하면서도 전문적으로 분석해 주세요.
                        [작성 규칙]
                        - 절대로 마크다운 구분선(---), 특수문자(###, **)를 사용하지 마시오.
                        - 단락 제목은 단순히 숫자와 번호로만 구분하시오.

                        1. 스윙 총평 및 칭찬
                        2. 단계별 메커니즘 진단 (어드레스, 백스윙, 임팩트 및 팔로우 스루)
                        3. 비거리 20m 늘리는 교정 팁 2가지
                        4. 추천 골프 교정용 연습 장비
                        """
                        
                        with open(golf_video_path, 'rb') as f:
                            golf_video_data = f.read()
                        
                        golf_response = model.generate_content([
                            golf_prompt, 
                            {"mime_type": "video/mp4", "data": golf_video_data}
                        ])
                        
                        st.session_state.golf_analysis = clean_report_text(golf_response.text)
                        st.session_state.golf_stage = 'analyzed'
                        
                    finally:
                        if os.path.exists(golf_video_path):
                            os.remove(golf_video_path)
                    
                    st.rerun()

    # 골프 스윙 결과 출력
    if 'golf_stage' in st.session_state and st.session_state.golf_stage == 'analyzed':
        st.divider()
        st.subheader("⛳ AI PGA 프로의 골프 스윙 진단 리포트")
        
        st.markdown(f"""
            <div style='background-color:#F0FDF4; padding:25px; border-radius:15px; border:1px solid #BBF7D0; line-height:1.8; color:#166534; font-size:15px;'>
                {st.session_state.golf_analysis}
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        if st.button("🎯 비거리 폭발! 추천 골프 연습용품 보러가기", use_container_width=True, key="golf_shop_btn"):
            st.session_state.golf_stage = 'shopping'
            st.rerun()

    if 'golf_stage' in st.session_state and st.session_state.golf_stage == 'shopping':
        st.subheader("🛒 형님을 위한 비거리 UP 골프 장비")
        g1, g2, g3 = st.columns(3)
        golf_items = [
            ("https://via.placeholder.com/300?text=Swing+Trainer", "🚀 비거리 20m 증가 스윙 연습기"),
            ("https://via.placeholder.com/300?text=Golf+Glove", "🖐️ 착감 폭발 양피 골프 장갑"),
            ("https://via.placeholder.com/300?text=Posture+Band", "🎗️ 슬라이스 방지 스윙 교정 밴드")
        ]
        for col, (img, name) in zip([g1, g2, g3], golf_items):
            with col:
                st.image(img)
                st.link_button(name, MY_REVENUE_LINK, use_container_width=True)
        
        st.success(f"형님, 이 장비들과 함께 필드에서 굿샷 하십시오! 최저가 클릭: {MY_REVENUE_LINK}")

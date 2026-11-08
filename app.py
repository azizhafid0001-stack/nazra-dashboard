import streamlit as st
import cv2
import numpy as np
import time
import av
from ultralytics import YOLO
import supervision as sv
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# ضبط إعدادات الصفحة
st.set_page_config(
    page_title="Nazra Smart Vision | نظام نظرة الذكي",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# تخصيص التصميم عبر CSS مخصص احترافي
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Tajawal:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cairo', 'Tajawal', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 35px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        color: white;
        margin-bottom: 25px;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .hero-desc {
        font-size: 1.1rem;
        color: #94A3B8;
        line-height: 1.6;
        max-width: 850px;
    }
    
    .badge-tag {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 14px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# واجهة الهبوط الرئيسية الاحترافية (Hero Section)
st.markdown("""
<div class="hero-container">
    <span class="badge-tag">نظام الرؤية الحاسوبية الميداني • AI Smart Hospitality</span>
    <div class="hero-title">نظام "نظرة" الذكي لإدارة ومراقبة الطاولات</div>
    <div class="hero-desc">
        منظومة ذكية تجمع بين خوارزميات الرؤية الحاسوبية اللحظية YOLOv8 والمعالجة السحابية الفورية لأتمتة خدمة المطاعم والمقاهي، حساب أوقات الجلوس، رصد الطاولات المتسخة، وتوجيه النادل بذكاء مع حماية كاملة لخصوصية العملاء.
    </div>
    <div style="margin-top: 15px; font-size: 0.95rem; color: #CBD5E1;">
        💼 <b>مطور النظام:</b> عزيز حفيظ | طالب بالمعهد المتخصص في مهن الطاقة المتجددة بطرفاية (تخصص: الطاقات المتجددة - شعبة طاقة الرياح)
    </div>
</div>
""", unsafe_allow_html=True)

# شريط التبويبات الرئيسي
tab_dashboard, tab_resume = st.tabs(["📊 لوحة المراقبة الحية (Dashboard)", "👤 السيرة الذاتية والملف المهني (Profile)"])

# تحميل نموذج YOLOv8
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8n.pt')

model = load_yolo_model()

with tab_dashboard:
    # بطاقات المؤشرات الإحصائية الأربعة المنفصلة
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("إجمالي الطاولات المراقبة", "1")
    col_m2.metric("🟢 طاولات جاهزة ونقية", "1")
    col_m3.metric("🔴 طاولات مشغولة", "0")
    col_m4.metric("🟠 طاولات تتطلب تنظيفاً", "0")

    st.markdown("---")

    col_video, col_controls = st.columns([1.8, 1.2])

    class SmartTableProcessor(VideoProcessorBase):
        def __init__(self):
            # مضلع الطاولة وسط الشاشة
            self.polygon = np.array([[30, 40], [610, 40], [610, 460], [30, 460]], np.int32)
            self.zone = sv.PolygonZone(polygon=self.polygon, triggering_anchors=[sv.Position.CENTER, sv.Position.BOTTOM_CENTER])
            # كادر أزرق للشخص الجالس
            self.person_annotator = sv.BoxAnnotator(thickness=3, color=sv.Color(255, 140, 0))
            # كادر ليموني / برتقالي للمواعين والكؤوس والقنينات
            self.dish_annotator = sv.BoxAnnotator(thickness=2, color=sv.Color(0, 165, 255))
            self.start_time = None
            self.last_seen_person = None
            self.last_infer_time = 0.0
            self.latest_detections = None

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (640, 480))
            current_time = time.time()

            # تسريع الأداء وتحليل الكوادر كل 0.25 ثانية
            if current_time - self.last_infer_time >= 0.25:
                self.last_infer_time = current_time
                # 0: person, 39: bottle, 41: cup, 42: fork, 43: knife, 44: spoon, 45: bowl
                results = model(img, imgsz=320, classes=[0, 39, 41, 42, 43, 44, 45], conf=0.18, verbose=False)[0]
                self.latest_detections = sv.Detections.from_ultralytics(results)

            dets = self.latest_detections
            has_person = False
            has_dishes = False

            if dets is not None and len(dets) > 0:
                in_zone = self.zone.trigger(dets)
                classes_in_zone = dets.class_id[in_zone]
                raw_person = 0 in classes_in_zone
                has_dishes = any(c in [39, 41, 42, 43, 44, 45] for c in classes_in_zone)

                # مهلة تأكيد الجلوس وتفادي الحركة السريعة
                if raw_person:
                    self.last_seen_person = current_time
                    has_person = True
                elif self.last_seen_person is not None and (current_time - self.last_seen_person) <= 3.0:
                    has_person = True

                # رسم الكادر الأزرق للشخص
                person_dets = dets[dets.class_id == 0]
                if len(person_dets) > 0:
                    img = self.person_annotator.annotate(scene=img, detections=person_dets)

                # رسم الكادر الليموني للمواعين والكؤوس
                dish_dets = dets[np.isin(dets.class_id, [39, 41, 42, 43, 44, 45])]
                if len(dish_dets) > 0:
                    img = self.dish_annotator.annotate(scene=img, detections=dish_dets)

            # منطق الألوان والمؤقت الزمني
            if has_person:
                if self.start_time is None:
                    self.start_time = current_time
                elapsed = int(current_time - self.start_time)

                if elapsed <= 5:
                    table_color = (0, 255, 255) # أصفر (تأكيد)
                    label_text = f"Confirming... ({elapsed}s/5s)"
                elif elapsed <= 30:
                    table_color = (0, 0, 255) # أحمر (مشغولة)
                    label_text = f"Occupied ({elapsed}s/30s)"
                else:
                    table_color = (0, 0, 255) # أحمر مع تنبيه النادل
                    label_text = f"WAITER NEEDED! ({elapsed}s)"
            elif has_dishes:
                self.start_time = None
                self.last_seen_person = None
                table_color = (0, 165, 255) # ليموني / برتقالي (تنظيف)
                label_text = "DIRTY - Needs Cleaning (Dishes/Bottles)"
            else:
                self.start_time = None
                self.last_seen_person = None
                table_color = (0, 255, 0) # أخضر (جاهزة ونقية)
                label_text = "FREE & CLEAN (0s)"

            # رسم كادر الطاولة ومستطيل العداد العلوي
            cv2.polylines(img, [self.polygon], isClosed=True, color=table_color, thickness=4)
            cv2.rectangle(img, (30, 0), (610, 40), (0, 0, 0), -1)
            cv2.putText(img, label_text, (40, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, table_color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    with col_video:
        st.subheader("📹 البث الميداني الحي المباشر")
        webrtc_streamer(
            key="nazra-real-stream",
            video_processor_factory=SmartTableProcessor,
            rtc_configuration=RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]}
                ]
            }),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

    with col_controls:
        st.subheader("🔔 تنبيهات الخدمة والتشغيل")
        st.info("🟢 **كادر أخضر (0s):** الطاولة فارغة ونقية وجاهزة للزبائن الجدد.")
        st.warning("🟡 **كادر أصفر (1-5 ثوانٍ):** رصد وجود شخص في طور التحقق والاستقرار.")
        st.error("🔴 **كادر أحمر (5-30 ثانية):** الطاولة مشغولة رسمياً ويتم احتساب وقت الجلوس.")
        st.error("🚨 **تنبيه عاجل (> 30 ثانية):** الزبون تجاوز 30 ثانية بدون طلب، خاص السرباي يجي لعندو!")
        st.warning("🟠 **كادر ليموني (برتقالي):** غادر الزبون وبقيت كؤوس/قنينات - الطاولة موسخة وتحتاج تنظيف فوراً.")
        st.info("🔵 **كادر أزرق:** يحيط بالشخص لتأكيد تمييز وجوده.")

        with st.expander("🛒 قائمة إرسال الطلبات للسرباي", expanded=True):
            items = st.multiselect("اختر الطلبات:", ["☕ قهوة إبريسو", "🥛 كابتشينو", "🫖 شاي مغربي", "🥐 كرواسون"])
            if st.button("إرسال فوري للنادل", use_container_width=True):
                if items:
                    st.success("✅ تم إرسال الإشعار بنجاح إلى شاشة الخدمة!")

with tab_resume:
    st.subheader("📄 السيرة الذاتية المهنية (CV & Portfolio)")
    
    c_res1, c_res2 = st.columns([1, 2])
    
    with c_res1:
        st.markdown("""
        ### **عزيز حفيظ | Aziz Hafid**
        **المستوى الأكاديمي:** طالب تقني متخصص  
        **المؤسسة:** المعهد المتخصص في مهن الطاقة المتجددة بطرفاية (ISMRT)  
        **الشعبة والتخصص:**  
        * الطاقات المتجددة (Énergies Renouvelables)  
        * **تخصص: طاقة الرياح (Énergie Éolienne)**  
        
        📍 *طرفاية، المملكة المغربية*
        """)
        st.link_button("🌐 زيارة حسابي على GitHub", "https://github.com/azizhafid0001-stack")

    with c_res2:
        st.markdown("""
        #### **النبذة المهنية (Professional Summary):**
        طالب في معهد الطاقات المتجددة بطرفاية، متخصص في مجال **الطاقات المتجددة (تخصص طاقة الرياح)**، وشغوف بتطوير وتطويع تقنيات **الذكاء الاصطناعي، إنترنت الأشياء (IoT)، والرؤية الحاسوبية (Computer Vision)** لخدمة المنشآت والمشاريع الميدانية. 
        
        أعمل على دمج المعارف الهندسية والتقنية الحديثة مع البرمجة التطبيقية لبناء حلول ذكية وعملية مثل نظام **"نظرة"** لأتمتة الفضاءات التجارية ورفع الكفاءة التشغيلية وترشيد استهلاك الموارد مع الحفاظ الصارم على خصوصية البيانات.
        
        ---
        #### **المهارات والتقنيات:**
        * **تقنيات الطاقة:** أنظمة طاقة الرياح (Eolien)، الشبكات الكهربائية، حساسات القياس والتحكم.
        * **الذكاء الاصطناعي:** YOLOv8 Object Detection, OpenCV, Supervision, Edge Computing.
        * **التطوير البرمجي:** Python, Streamlit Framework, WebRTC Live Video Processing, GitHub.
        """)

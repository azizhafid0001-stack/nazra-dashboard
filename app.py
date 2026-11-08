import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(
    page_title="Nazra Live Dashboard",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Nazra Live Dashboard - نظام المراقبة الذكي للطاولات")
st.markdown("إعداد: **عزيز حفيظ** | **المعهد المتخصص في مهن الطاقة المتجددة بطرفاية**")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# حالة الطاولة والمؤقت في الذاكرة
class TableTracker:
    def __init__(self):
        self.start_time = None
        self.last_seen_person = None
        self.status = "🟢 فارغة ونقية"
        self.elapsed = 0
        self.is_occupied = False
        self.needs_cleaning = False

tracker = TableTracker()

# إعداد منطقة الطاولة (Polygon ROI) في وسط الكاميرا
polygon = np.array([
    [50, 40],
    [590, 40],
    [590, 440],
    [50, 440]
], np.int32)

zone = sv.PolygonZone(polygon=polygon, triggering_anchors=[sv.Position.CENTER, sv.Position.BOTTOM_CENTER])
box_annotator = sv.BoxAnnotator(thickness=2)

st.markdown("---")

# إحصائيات علوية
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
m1 = col_m1.empty()
m2 = col_m2.empty()
m3 = col_m3.empty()
m4 = col_m4.empty()

def update_metrics(occupied=False, dirty=False):
    free = 1 if (not occupied and not dirty) else 0
    occ = 1 if occupied else 0
    dirt = 1 if dirty else 0
    m1.metric("إجمالي الطاولات", "1")
    m2.metric("🟢 فارغة ونقية", f"{free}")
    m3.metric("🔴 مشغولة بالزبائن", f"{occ}")
    m4.metric("🟠 تحتاج تنظيف (موسخة)", f"{dirt}")

update_metrics()

st.markdown("---")
col_video, col_info = st.columns([1.8, 1.2])

with col_info:
    st.subheader("🔔 حالة الطاولة والتنبيهات المباشرة")
    status_alert = st.empty()
    timer_text = st.empty()
    details_box = st.empty()

# معالجة إطارات الفيديو ورسم الكوادر
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, (640, 480))
    current_time = time.time()

    # كشف الأشخاص (0) والأواني/الكؤوس/القراعي (39: bottle, 41: cup, 42: fork, 43: knife, 44: spoon, 45: bowl)
    results = model(img, imgsz=320, classes=[0, 39, 41, 42, 43, 44, 45], conf=0.20, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # التحقق من العناصر داخل كادر الطاولة
    in_zone = zone.trigger(detections)
    classes_in_zone = detections.class_id[in_zone]

    raw_has_person = 0 in classes_in_zone
    has_dishes = any(c in [39, 41, 42, 43, 44, 45] for c in classes_in_zone)

    # فترة سماح 3.5 ثواني لمنع التقطع عند حركة الشخص
    if raw_has_person:
        tracker.last_seen_person = current_time
        has_person = True
    elif tracker.last_seen_person is not None and (current_time - tracker.last_seen_person) <= 3.5:
        has_person = True
    else:
        has_person = False

    # تحديد الألوان والحالة والعداد
    if has_person:
        if tracker.start_time is None:
            tracker.start_time = current_time

        tracker.elapsed = int(current_time - tracker.start_time)
        tracker.is_occupied = True
        tracker.needs_cleaning = False

        if tracker.elapsed <= 5:
            box_color = (0, 255, 255) # 🟡 أصفر (طور التأكيد)
            label_text = f"CONFIRMING ({tracker.elapsed}s/5s)"
            tracker.status = "🟡 شخص جالس - طور التأكيد"
        elif tracker.elapsed <= 30:
            box_color = (0, 0, 255)   # 🔴 أحمر (مشغولة)
            label_text = f"OCCUPIED ({tracker.elapsed}s/30s)"
            tracker.status = f"🔴 الطاولة مشغولة ({tracker.elapsed}s)"
        else:
            box_color = (0, 0, 255)   # 🔴 أحمر (تنبيه السرباي)
            label_text = f"WAITER NEEDED! ({tracker.elapsed}s)"
            tracker.status = f"🚨 تنبيه نادل! فات الزبون {tracker.elapsed}s"

    elif has_dishes:
        tracker.start_time = None
        tracker.last_seen_person = None
        tracker.is_occupied = False
        tracker.needs_cleaning = True
        tracker.elapsed = 0
        box_color = (0, 140, 255)    # 🟠 ليموني / برتقالي (تحتاج تنظيف)
        label_text = "DIRTY - DISHES DETECTED!"
        tracker.status = "🟠 غير نظيفة (بقاو المواعن)"

    else:
        tracker.start_time = None
        tracker.last_seen_person = None
        tracker.is_occupied = False
        tracker.needs_cleaning = False
        tracker.elapsed = 0
        box_color = (0, 255, 0)      # 🟢 خضر (فارغة ونقية)
        label_text = "FREE & CLEAN"
        tracker.status = "🟢 فارغة ونقية"

    # 1. رسم الكادر الزرق فوق أي شخص يتم رصده (BBox)
    person_detections = detections[detections.class_id == 0]
    if len(person_detections) > 0:
        for xyxy in person_detections.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2) # كادر أزرق للشخص
            cv2.putText(img, "Person", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # 2. رسم كادر المواعين باللون البرتقالي
    dish_detections = detections[detections.class_id != 0]
    if len(dish_detections) > 0:
        for xyxy in dish_detections.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)

    # 3. رسم كادر الطاولة الكبير بالألوان (خضر / صفر / حمر / ليموني)
    cv2.polylines(img, [polygon], isClosed=True, color=box_color, thickness=4)
    cv2.rectangle(img, (50, 5), (590, 40), (0, 0, 0), -1)
    cv2.putText(img, label_text, (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_video:
    st.subheader("📹 بث المراقبة المباشر (Live Video)")
    # إعدادات WebRTC مع خوادم Google لضمان العمل على أي متصفح وشبكة
    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]
    })

    webrtc_ctx = webrtc_streamer(
        key="nazra-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# تحديث معلومات الواجهة الجانبية طالما الفيديو مشغل
while webrtc_ctx.state.playing:
    update_metrics(tracker.is_occupied, tracker.needs_cleaning)

    if tracker.is_occupied:
        if tracker.elapsed <= 5:
            status_alert.warning(f"⏳ {tracker.status}")
            details_box.info("الزبون جلس للتو، النظام يقوم بالتأكيد...")
        elif tracker.elapsed <= 30:
            status_alert.error(f"👤 {tracker.status}")
            details_box.write("الزبون مستقر في الطاولة والخدمة قيد الانتظار.")
        else:
            status_alert.error(f"🚨 **تنبيه عاجل للنادل!**\n\nتجاوزت الطاولة 30 ثانية دون تقديم الطلب.")
            details_box.error("توجه فوراً إلى طاولة المجسم!")
        timer_text.metric("⏱️ العداد الزمني للجلوس", f"{tracker.elapsed} ثانية")

    elif tracker.needs_cleaning:
        status_alert.warning(f"🧹 {tracker.status}")
        timer_text.empty()
        details_box.warning("توجد أوانٍ أو كؤوس متروكة! يرجى إرسال عامل النظافة لتجهيز الطاولة.")

    else:
        status_alert.success(f"✨ {tracker.status}")
        timer_text.empty()
        details_box.info("الطاولة شاغرة ونظيفة تماماً وبانتظار زبائن جدد.")

    time.sleep(0.5)

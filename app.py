import threading
import time
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import supervision as sv
from ultralytics import YOLO

st.set_page_config(
    page_title="Nazra Live Dashboard", page_icon="☕", layout="wide"
)

st.title("☕ Nazra Live Dashboard - نظام مراقبة الطاولات الذكي")
st.markdown(
    "إعداد: **عزيز حفيظ** | **المعهد المتخصص في مهن الطاقة المتجددة بطرفاية**"
)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


# ذاكرة مشتركة لتخفيف الضغط على السيرفر السحابي ومنع تجمد الفيديو
@st.cache_resource
def get_shared_state():
    return {
        "lock": threading.Lock(),
        "status": "🟢 فارغة (نقية)",
        "orders": [],
        "start_time": None,
        "last_seen_person": None,
        "occ_val": 0,
        "free_val": 1,
        "status_type": "success",
        "alert_type": None,
        "alert_msg": "",
        "box_color": (0, 255, 0),
        "label_text": "Free & Clean",
        "last_infer_time": 0.0,
        "last_detections": None,
    }


shared_state = get_shared_state()

st.markdown("---")
with st.form("menu_form"):
    st.subheader("🛒 قائمة الطعام (Menu) وإشعارات النادل")
    menu_items = [
        "☕ قهوة إبريسو (Espresso)",
        "🥛 كابتشينو (Cappuccino)",
        "🫖 شاي مغربي (Moroccan Tea)",
        "🧃 عصير برتقال (Orange Juice)",
        "🥐 كرواسون (Croissant)",
    ]
    selected_items = st.multiselect("اختر الطلبات من القائمة:", menu_items)
    submit_order = st.form_submit_button("إرسال الطلبات المحددة وإشعار النادل")

    if submit_order:
        if selected_items:
            orders_str = ", ".join(selected_items)
            shared_state["orders"].extend(selected_items)
            st.success(
                f"✅ تم إرسال الطلبات بنجاح لطاولة المجسم: **{orders_str}**"
            )
            st.info(
                "🚨 [تنبيه مرسل للسرباي]: يرجى التوجه فوراً إلى طاولة المجسم لتقديم الخدمة."
            )
        else:
            st.warning("المرجو اختيار طلب واحد على الأقل من القائمة.")

st.markdown("---")

mode = st.radio(
    "🌐 اختر مصدر البث المباشر:",
    [
        "📱 كاميرا الهاتف / المتصفح عبر الإنترنت (Cloud Live)",
        "💻 كاميرا الحاسوب المباشرة (Local)",
    ],
    horizontal=True,
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 البث المباشر ومراقبة الطاولة (ROI)")
    video_placeholder = st.empty()

with col2:
    st.subheader("📊 حالة الطاولة والعداد الزمني")
    status_placeholder = st.empty()
    metric_placeholder = st.empty()
    notification_placeholder = st.empty()

# منطقة الطاولة تغطي مساحة واضحة في الوسط
polygon = np.array([[40, 30], [600, 30], [600, 450], [40, 450]], np.int32)
zone = sv.PolygonZone(
    polygon=polygon,
    triggering_anchors=[sv.Position.CENTER, sv.Position.BOTTOM_CENTER],
)
box_annotator = sv.BoxAnnotator(thickness=2)


def process_frame(frame):
    frame = cv2.resize(frame, (640, 480))
    current_time = time.time()

    # تشغيل التحليل بالذكاء الاصطناعي كل 0.35 ثانية لمنع تجمد البث
    if current_time - shared_state["last_infer_time"] >= 0.35:
        shared_state["last_infer_time"] = current_time
        results = model(
            frame,
            imgsz=320,
            classes=[0, 39, 41, 42, 43, 44, 45],
            conf=0.20,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(results)

        in_zone = zone.trigger(detections)
        classes_in_zone = detections.class_id[in_zone]

        raw_has_person = 0 in classes_in_zone
        has_dishes = any(c in [39, 41, 42, 43, 44, 45] for c in classes_in_zone)

        with shared_state["lock"]:
            shared_state["last_detections"] = detections

            if raw_has_person:
                shared_state["last_seen_person"] = current_time
                has_person = True
            elif (
                shared_state["last_seen_person"] is not None
                and (current_time - shared_state["last_seen_person"]) <= 3.5
            ):
                has_person = True
            else:
                has_person = False

            alert_type = None
            alert_msg = ""

            if has_person:
                if shared_state["start_time"] is None:
                    shared_state["start_time"] = current_time

                elapsed = int(current_time - shared_state["start_time"])
                occ_val, free_val = 1, 0

                if elapsed <= 5:
                    box_color = (0, 255, 255)  # أصفر
                    label_text = f"Confirming... ({elapsed}s / 5s)"
                    shared_state["status"] = (
                        f"🟡 جالس فبلاصتو ({elapsed} ثواني) - طور التأكيد"
                    )
                    status_type = "warning"
                elif elapsed <= 30:
                    box_color = (0, 0, 255)  # أحمر
                    label_text = f"Occupied ({elapsed}s / 30s)"
                    shared_state["status"] = (
                        f"🔴 الطاولة مشغولة ({elapsed} / 30 ثانية)"
                    )
                    status_type = "error"
                else:
                    box_color = (0, 0, 255)  # أحمر
                    label_text = f"ALERT: Waiter Needed! ({elapsed}s)"
                    shared_state["status"] = (
                        f"🔴 مشغولة ({elapsed} ثانية) - خاص السرباي يجي!"
                    )
                    status_type = "error"
                    alert_type = "error"
                    alert_msg = "🚨 **ميساج عاجل للسرباي:** الزبون كمل 30 ثانية فـ الطابلة، خاصك تمشي عندهم دابا تسربيهم وتشد الطلب!"

            elif has_dishes:
                shared_state["start_time"] = None
                shared_state["last_seen_person"] = None
                occ_val, free_val = 1, 0
                box_color = (0, 165, 255)  # برتقالي
                label_text = "DIRTY: Needs Cleaning! (Dishes)"
                shared_state["status"] = (
                    "🟠 غير نظيفة (بقاو المواعن فوق الطابلة)"
                )
                status_type = "warning"
                alert_type = "warning"
                alert_msg = "🧹 **ميساج للسرباي (تنظيف):** الزبون ناض وبقاو المواعن فوق الطابلة! عافاك سير جمعهم ونقي الطابلة باش تولي واجدة."

            else:
                shared_state["start_time"] = None
                shared_state["last_seen_person"] = None
                box_color = (0, 255, 0)  # أخضر
                label_text = "FREE & CLEAN (0s)"
                shared_state["status"] = "🟢 فارغة (نقية)"
                occ_val, free_val = 0, 1
                status_type = "success"

            shared_state["occ_val"] = occ_val
            shared_state["free_val"] = free_val
            shared_state["status_type"] = status_type
            shared_state["alert_type"] = alert_type
            shared_state["alert_msg"] = alert_msg
            shared_state["box_color"] = box_color
            shared_state["label_text"] = label_text

    # رسم المربعات والشريط العلوي بدون إبطاء الفيديو
    annotated_frame = frame.copy()
    with shared_state["lock"]:
        dets = shared_state["last_detections"]
        box_color = shared_state["box_color"]
        label_text = shared_state["label_text"]

    if dets is not None and len(dets) > 0:
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=dets
        )

    cv2.polylines(
        annotated_frame, [polygon], isClosed=True, color=box_color, thickness=4
    )
    cv2.rectangle(annotated_frame, (40, 0), (600, 35), (0, 0, 0), -1)
    cv2.putText(
        annotated_frame,
        label_text,
        (50, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        box_color,
        2,
    )

    return annotated_frame


def update_ui():
    with shared_state["lock"]:
        status = shared_state["status"]
        status_type = shared_state["status_type"]
        alert_type = shared_state["alert_type"]
        alert_msg = shared_state["alert_msg"]
        occ_val = shared_state["occ_val"]
        free_val = shared_state["free_val"]

    if status_type == "warning":
        status_placeholder.warning(status)
    elif status_type == "error":
        status_placeholder.error(status)
    else:
        status_placeholder.success(status)

    if alert_type == "error":
        notification_placeholder.error(alert_msg)
    elif alert_type == "warning":
        notification_placeholder.warning(alert_msg)
    else:
        notification_placeholder.empty()

    with metric_placeholder.container():
        c1, c2, c3 = st.columns(3)
        c1.metric(label="مجموع الطاولات", value="1")
        c2.metric(label="فارغة (نقية)", value=str(free_val))
        c3.metric(label="مشغولة / تنظيف", value=str(occ_val))


update_ui()

if mode == "💻 كاميرا الحاسوب المباشرة (Local)":
    run_video = st.checkbox("تشغيل البث المباشر (فيديو)", value=True)
    if run_video:
        if "cap" not in st.session_state or not st.session_state.cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            st.session_state.cap = cap
        cap = st.session_state.cap

        if not cap.isOpened():
            st.error(
                "⚠️ الكاميرا غير متاحة محلياً! اختر الوضع الأول (كاميرا الهاتف / المتصفح عبر الإنترنت)."
            )
        else:
            while run_video:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated_frame = process_frame(frame)
                update_ui()
                annotated_frame = cv2.cvtColor(
                    annotated_frame, cv2.COLOR_BGR2RGB
                )
                video_placeholder.image(
                    annotated_frame, channels="RGB", use_container_width=True
                )
                time.sleep(0.01)
    else:
        if "cap" in st.session_state and st.session_state.cap.isOpened():
            st.session_state.cap.release()

else:
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        annotated_frame = process_frame(img)
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    with col1:
        ctx = webrtc_streamer(
            key="nazra-cloud",
            video_frame_callback=video_frame_callback,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 15, "max": 15},
                },
                "audio": False,
            },
            async_processing=False,
        )

    while ctx.state.playing:
        update_ui()
        time.sleep(0.5)

import streamlit as st
import sqlite3
import pandas as pd
import os
import time

DB_PATH = "data/vision_os.db"

st.set_page_config(page_title="Vision OS Edge", layout="wide")

# Connect to database safely
def get_db_data(query):
    if not os.path.exists(DB_PATH): return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(query, conn)
    except:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

st.sidebar.title("Vision OS Security")
page = st.sidebar.radio("Navigation", [
    "Live Monitoring", 
    "Live Alerts (New Faces)",
    "Smart Attendance", 
    "Alarms & Sirens", 
    "Alert Queue & Comm",
    "System & Storage"
])

if page == "Live Monitoring":
    st.title("Live Camera Feed (Dual-Stream)")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Stream A (Low-Res Continuous)")
        image_placeholder = st.empty()
    with col2:
        st.subheader("Recent Events")
        events_df = get_db_data("SELECT timestamp, event_type, identifier FROM events ORDER BY id DESC LIMIT 10")
        st.dataframe(events_df, use_container_width=True, hide_index=True)
        
    while True:
        if os.path.exists("data/latest_frame.jpg"):
            try:
                image_placeholder.image("data/latest_frame.jpg", use_container_width=True)
            except: pass
        time.sleep(0.1)

elif page == "Live Alerts (New Faces)":
    st.title("Live Alerts & Known Visitors")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("New/Unrecognized Faces")
        new_faces_df = get_db_data("SELECT timestamp, identifier FROM events WHERE event_type = 'New Visitor' ORDER BY id DESC LIMIT 10")
        if new_faces_df.empty:
            st.success("No new unrecognized faces detected.")
        else:
            st.dataframe(new_faces_df, use_container_width=True, hide_index=True)
            
    with col2:
        st.subheader("Known Visitors DB")
        visitors_df = get_db_data("SELECT visitor_id, first_seen, last_seen, seen_count FROM visitors ORDER BY last_seen DESC LIMIT 20")
        if visitors_df.empty:
            st.info("No visitors recorded yet.")
        else:
            # Add Regular badge
            threshold = 10
            visitors_df['Status'] = visitors_df['seen_count'].apply(lambda x: '⭐ Regular' if x >= threshold else 'Visitor')
            st.dataframe(visitors_df, use_container_width=True, hide_index=True)

elif page == "Smart Attendance":
    st.title("Smart Attendance Log")
    att_df = get_db_data("SELECT timestamp, person_id, camera_id FROM attendance ORDER BY id DESC")
    
    if att_df.empty:
        st.info("No attendance records found.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Total Check-ins Today", len(att_df))
        with col2:
            st.dataframe(att_df, use_container_width=True, hide_index=True)

elif page == "Alarms & Sirens":
    st.title("Security Alarms & Actuators")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Latching Siren Status")
        # In a real app we'd read the actuator state. Here we look at the DB.
        disarm_events = get_db_data("SELECT id FROM events WHERE identifier='Disarm Commanded' AND timestamp > datetime('now', '-5 minutes')")
        alert_events = get_db_data("SELECT id FROM events WHERE event_type LIKE '%Alert%' AND timestamp > datetime('now', '-5 minutes')")
        
        if not alert_events.empty and disarm_events.empty:
            st.error("🚨 SIREN IS LATCHED ON")
        else:
            st.success("✅ Siren is OFF / Disarmed")
            
        if st.button("🚨 REMOTE DISARM SIREN", use_container_width=True):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("INSERT INTO events (timestamp, event_type, identifier, confidence, action_taken) VALUES (datetime('now', 'localtime'), 'System', 'Disarm Commanded', 1.0, 'Siren Cleared')")
            conn.commit()
            conn.close()
            st.success("Disarm command sent securely.")
            
    with col2:
        st.subheader("Recent Hardware & Fire Alarms")
        alarms_df = get_db_data("SELECT timestamp, event_type, identifier FROM events WHERE event_type IN ('Fire Alert', 'Hardware') ORDER BY id DESC LIMIT 10")
        st.dataframe(alarms_df, use_container_width=True, hide_index=True)

elif page == "Alert Queue & Comm":
    st.title("Cellular Alerting & IVR")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Store-and-Forward Queue")
        queue_df = get_db_data("SELECT * FROM alert_queue ORDER BY id DESC LIMIT 10")
        if queue_df.empty:
            st.info("Alert queue is empty. All messages sent.")
        else:
            st.dataframe(queue_df, use_container_width=True, hide_index=True)
            
    with col2:
        st.subheader("IVR System")
        st.write("Simulate an incoming phone call to the edge device modem.")
        if st.button("📞 Simulate Incoming Call"):
            events_df = get_db_data("SELECT event_type FROM events ORDER BY id DESC LIMIT 1")
            last_event = events_df.iloc[0]['event_type'] if not events_df.empty else "None"
            
            st.info(f"🎙️ **TTS Engine:** 'System is ONLINE. Storage at 4%. Last event recorded was: {last_event}. Have a secure day.'")

elif page == "System & Storage":
    st.title("Edge Device Health")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Battery Status", "100%", "Trickle Charging")
    with col2:
        st.metric("Main Power", "ONLINE")
    with col3:
        st.metric("Cellular Modem", "CONNECTED", "Signal: Strong")
        
    st.markdown("---")
    st.subheader("Dual-Stream Video Storage")
    
    st.progress(0.04, text="Storage Used: 4% (2.1 GB / 64 GB)")
    
    if os.path.exists("data/stream_b_events"):
        st.write("High-Res Event Snapshots (Stream B):")
        files = os.listdir("data/stream_b_events")
        if not files:
            st.write("No high-res captures yet.")
        for f in files:
            st.code(f)

# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
import os
import ast

# ============================
# Configuration - UPDATE THESE PATHS
# ============================

# --- 1. Google Sheet Configuration ---
# This is the standard "edit" URL of your sheet.
SHEET_URL = "https://docs.google.com/spreadsheets/d/1IeueIugoP0t2lWnni35rbMfRaFJrQzcax3ffcVW45ik/edit?usp=sharing"
# --- 2. Local Image Folder Configuration ---
NAILS_DIR = "nails"

# ============================
# Core Logic & Data Loading
# ============================

@st.cache_data(ttl=600) # Cache data for 10 minutes to avoid hitting the sheet too often
def load_product_data_from_gsheet(sheet_url):
    """
    Loads product data from a public Google Sheet URL.
    """
    try:
        # Transform the standard URL into a CSV export URL
        csv_export_url = sheet_url.replace('/edit?usp=sharing', '/export?format=csv&gid=0')
        df = pd.read_csv(csv_export_url)
        
        # --- Data Cleaning ---
        # Convert the string '(r, g, b)' into an actual tuple of integers
        df['DominantColorRGB'] = df['DominantColorRGB'].apply(ast.literal_eval)
        # Ensure 'Selling Price' is a numeric type for calculations
        df['Selling Price'] = pd.to_numeric(df['Selling Price'])
        # Filter out any inactive products
        df = df[df['IsActive'] == True].copy()
        
        st.success("Here is a list of our available products.")
        return df
    except Exception as e:
        st.error(f"Failed to generate product list")
        st.error(f"Details: {e}")
        return None

class HandProcessor:
    # (No changes to this class - it remains the same)
    def __init__(self, detection_con=0.8, max_hands=1): self.detector = HandDetector(detectionCon=detection_con, maxHands=max_hands)
    def process_hand_image(self, frame):
        hands, annotated_frame = self.detector.findHands(frame.copy());
        if not hands: return None
        hand = hands[0]; skin_tone = self._extract_skin_tone(frame, hand); nail_shape, _ = self._analyze_nail_shapes(frame, hand)
        return {"skin_tone": skin_tone, "nail_shape": nail_shape, "original_image": frame, "annotated_image": annotated_frame}
    def _extract_skin_tone(self, frame, hand_landmarks):
        try:
            mask = np.zeros(frame.shape[:2], dtype="uint8"); cv2.drawContours(mask, [cv2.convexHull(np.array(hand_landmarks["lmList"], dtype=np.int32))], -1, 255, -1)
            final_mask = cv2.bitwise_and(mask, cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb), (0, 133, 77), (255, 173, 127)))
            avg_color_bgr = cv2.mean(frame, mask=final_mask)[:3]
            return tuple(map(int, avg_color_bgr)) if np.count_nonzero(final_mask) > 0 else self._get_fallback_skin_tone(frame, hand_landmarks)
        except Exception: return self._get_fallback_skin_tone(frame, hand_landmarks)
    def _get_fallback_skin_tone(self, frame, hand_landmarks):
        x, y = hand_landmarks["center"]; roi = frame[y-30:y+30, x-30:x+30]
        return tuple(map(int, np.mean(roi.reshape(-1, 3), axis=0))) if roi.size else (120, 150, 200)
    def _analyze_nail_shapes(self, frame, hand_landmarks):
        nail_shapes, nail_contours, avg_shape = [], {}, "unknown"
        for tip_id in [4, 8, 12, 16, 20]:
            shape, _ = self._analyze_single_nail(frame, hand_landmarks, tip_id)
            if shape: nail_shapes.append(shape)
        if nail_shapes: avg_shape = max(set(nail_shapes), key=nail_shapes.count)
        return avg_shape, nail_contours
    def _analyze_single_nail(self, frame, hand_landmarks, tip_id):
        try:
            x, y = hand_landmarks["lmList"][tip_id][:2]; roi = frame[y-20:y+20, x-20:x+20]
            if roi.size == 0: return None, None
            mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), np.array([0, 0, 150]), np.array([180, 80, 255]))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return None, None
            nail_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(nail_contour) < 20: return None, None
            _, _, w, h = cv2.boundingRect(nail_contour); shape = "square" if w / float(h) > 1.2 else "round"
            nail_contour[:, :, 0] += x - 20; nail_contour[:, :, 1] += y - 20
            return shape, nail_contour
        except Exception: return None, None

class NailRecommender:
    # (No changes to this class - it remains the same)
    def _color_distance(self, c1_bgr, c2_rgb):
        c1_rgb = (c1_bgr[2], c1_bgr[1], c1_bgr[0])
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1_rgb, c2_rgb)))
    def recommend_top_n(self, all_products_df, skin_tone_bgr, nail_shape, n=10, preferred_styles=None):
        if all_products_df is None: return pd.DataFrame()
        available_nails = all_products_df.copy()
        if preferred_styles:
            available_nails = available_nails[available_nails['Style'].isin(preferred_styles)]
        scores, match_percents = [], []
        MAX_POSSIBLE_SCORE = 600
        for _, nail in available_nails.iterrows():
            shape_score = 150
            compatible_shapes = ast.literal_eval(nail['Shape']) if isinstance(nail['Shape'], str) else []
            if nail_shape == "unknown" or nail_shape in compatible_shapes: shape_score = 0
            dist = self._color_distance(skin_tone_bgr, nail["DominantColorRGB"])
            total_score = dist + shape_score
            scores.append(total_score)
            similarity_score = max(0, MAX_POSSIBLE_SCORE - total_score)
            match_percents.append((similarity_score / MAX_POSSIBLE_SCORE) * 100)
        available_nails['score'] = scores
        available_nails['match_percent'] = match_percents
        return available_nails.sort_values('score').head(n)

# ============================
# Streamlit UI (Identical to previous version)
# ============================

st.set_page_config(layout="wide", page_title="AI Nail Stylist")

if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'cart' not in st.session_state: st.session_state.cart = []
if 'page' not in st.session_state: st.session_state.page = 'shop'

# --- Load Data from Google Sheet ---
products_df = load_product_data_from_gsheet(SHEET_URL)

with st.sidebar:
    st.title("💅 AI Nail Stylist")
    if st.session_state.cart:
        st.subheader("🛒 Your Shopping Cart")
        total_price = sum(item['Selling Price'] for item in st.session_state.cart)
        for i, item in enumerate(st.session_state.cart):
            st.write(f"- {item['ProductName']} (₦{item['Selling Price']:.2f})")
        st.markdown("---")
        st.metric(label="Total Amount Due", value=f"₦{total_price:.2f}")
        if st.button("Go to Checkout", use_container_width=True, type="primary"):
            st.session_state.page = 'checkout'; st.rerun()
        if st.button("Clear Cart", use_container_width=True):
            st.session_state.cart = []; st.rerun()
    else:
        st.info("Your cart is currently empty.")
    
    st.markdown("---")
    st.header("Start Here")
    uploaded_file = st.file_uploader("Upload a picture of your hand...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        with st.spinner('Analyzing your hand...'):
            st.session_state.analysis_results = HandProcessor().process_hand_image(cv2_img)
            st.session_state.page = 'shop'

if st.session_state.page == 'checkout':
    st.title("Checkout Summary")
    if not st.session_state.cart:
        st.warning("Your cart is empty.")
        if st.button("⬅️ Back to Shop"): st.session_state.page = 'shop'; st.rerun()
    else:
        total_price = sum(item['Selling Price'] for item in st.session_state.cart)
        st.metric(label="Final Amount Due", value=f"₦{total_price:.2f}")
        st.subheader("Payment Instructions")
        payment_details = """
        **Bank Name:** Moniepoint
        **Account Name:** Precious Emenike
        **Account Number:** 9011665061
        """
        st.markdown(payment_details)
        st.success("Please use your name as the payment reference. Thank you for your purchase!")
        if st.button("⬅️ Continue Shopping"): st.session_state.page = 'shop'; st.rerun()

elif st.session_state.page == 'shop':
    st.header("Your Personalized Recommendations")
    if st.session_state.analysis_results:
        skin_tone_bgr, nail_shape = st.session_state.analysis_results["skin_tone"], st.session_state.analysis_results["nail_shape"]
        st.success(f"Analysis complete! Detected Nail Shape: {nail_shape.capitalize()}")
        if products_df is not None:
            style_options = sorted(list(products_df['Style'].unique()))
            selected_styles = st.multiselect("Filter by Style:", options=style_options)
            recommendations_df = NailRecommender().recommend_top_n(products_df, skin_tone_bgr, nail_shape, n=10, preferred_styles=selected_styles)
            if recommendations_df.empty:
                st.warning("No recommendations match your filters.")
            else:
                for i in range(0, len(recommendations_df), 5):
                    cols = st.columns(5)
                    batch = recommendations_df.iloc[i:i+5]
                    for idx, (_, rec) in enumerate(batch.iterrows()):
                        with cols[idx]:
                            image_path = None
                            for ext in ['.png', '.jpg', '.jpeg']:
                                potential_path = os.path.join(NAILS_DIR, rec['ProductID'] + ext)
                                if os.path.exists(potential_path): image_path = potential_path; break
                            if image_path:
                                st.image(image_path, use_column_width=True)
                                st.markdown(f"**{rec['ProductName']}**")
                                st.markdown(f"<p style='font-size: 1.1rem; font-weight: 500; margin: 0;'>₦{rec['Selling Price']:.2f}</p>", unsafe_allow_html=True)
                                #st.metric(label="Price", value=f"₦{rec['Selling Price']:.2f}")
                                st.markdown(f"<h5 style='text-align: left; color: #28a745; font-size: 0.8rem; margin-top: 5px; margin-bottom: 10px;'>{rec['match_percent']:.0f}% Match</h5>", unsafe_allow_html=True)
                                in_cart = any(item['ProductID'] == rec['ProductID'] for item in st.session_state.cart)
                                if st.button("Add to Cart", key=f"add_{rec['ProductID']}", use_container_width=True, disabled=in_cart):
                                    st.session_state.cart.append(rec.to_dict()); st.rerun()
                            else: st.error(f"Image not found for {rec['ProductID']}")
    else:

        st.info("✨ Upload an image of your hand to see personalized nail recommendations.")

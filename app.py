from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.impute._base")


MODEL_PATH = Path("best_network_anomaly_model.pkl")
TEST_DATA_PATH = Path("testing_data.csv")

RAW_COLUMNS = [
    "ts",
    "uid",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "local_orig",
    "local_resp",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "label",
]

NUMERIC_COLUMNS = [
    "ts",
    "id.orig_p",
    "id.resp_p",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]


st.set_page_config(
    page_title="MLC Network Detection System",
    layout="wide",
)


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    for col in ["service", "history", "orig_bytes", "resp_bytes"]:
        data[f"{col}_missing"] = data[col].isna().astype(int)

    ts = pd.to_datetime(data["ts"], unit="s", errors="coerce")
    data["hour"] = ts.dt.hour
    data["dayofweek"] = ts.dt.dayofweek

    data["total_bytes"] = data["orig_bytes"].fillna(0) + data["resp_bytes"].fillna(0)
    data["total_pkts"] = data["orig_pkts"].fillna(0) + data["resp_pkts"].fillna(0)
    data["bytes_per_pkt"] = data["total_bytes"] / data["total_pkts"].replace(0, np.nan)
    data["orig_resp_byte_ratio"] = data["orig_bytes"] / data["resp_bytes"].replace(0, np.nan)
    data["orig_resp_pkt_ratio"] = data["orig_pkts"] / data["resp_pkts"].replace(0, np.nan)

    data["orig_port_is_well_known"] = (data["id.orig_p"] <= 1024).astype(int)
    data["resp_port_is_well_known"] = (data["id.resp_p"] <= 1024).astype(int)

    data["orig_ip_prefix"] = data["id.orig_h"].astype(str).str.extract(r"^(\d+\.\d+)", expand=False)
    data["resp_ip_prefix"] = data["id.resp_h"].astype(str).str.extract(r"^(\d+\.\d+)", expand=False)

    return data.drop(columns=["uid", "label", "id.orig_h", "id.resp_h", "ts"], errors="ignore")


def prepare_model_features(data: pd.DataFrame) -> pd.DataFrame:
    if {"ts", "id.orig_h", "id.resp_h"}.issubset(data.columns):
        return engineer_features(data)
    return data.drop(columns=["traffic_type", "label", "uid"], errors="ignore").copy()


def normalize_prediction(value: Any) -> tuple[str, str]:
    raw = value.item() if hasattr(value, "item") else value

    if isinstance(raw, str):
        label = raw.strip()
        if label.lower() in {"1", "malicious", "anomaly", "attack"}:
            return "Malicious", "malicious"
        if label.lower() in {"0", "benign", "normal"}:
            return "Benign", "benign"
        return label, "unknown"

    if int(raw) == 1:
        return "Malicious", "malicious"
    return "Benign", "benign"


def get_prediction_scores(model: Any, features: pd.DataFrame) -> pd.DataFrame | None:
    if not hasattr(model, "predict_proba"):
        return None

    try:
        probabilities = model.predict_proba(features)[0]
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classes = getattr(model.named_steps.get("model"), "classes_", None)
        if classes is None:
            classes = list(range(len(probabilities)))

        rows = []
        for cls, probability in zip(classes, probabilities):
            label, _ = normalize_prediction(cls)
            rows.append({"label": label, "confidence": float(probability)})
        return pd.DataFrame(rows)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_model() -> tuple[Any | None, str | None]:
    if not MODEL_PATH.exists():
        return None, f"Model file not found: {MODEL_PATH}"

    try:
        return joblib.load(MODEL_PATH), None
    except Exception as joblib_error:
        try:
            with MODEL_PATH.open("rb") as model_file:
                return pickle.load(model_file), None
        except Exception as pickle_error:
            message = (
                "The saved model could not be loaded. This usually happens when the "
                "model was saved with a different scikit-learn version.\n\n"
                f"Joblib error: {joblib_error}\n\n"
                f"Pickle error: {pickle_error}\n\n"
                "Fix option 1: install the training version, for example "
                "`pip install scikit-learn==1.6.1`.\n\n"
                "Fix option 2: re-run the notebook in this environment and save a new "
                "`best_network_anomaly_model.pkl`."
            )
            return None, message


@st.cache_data(show_spinner=False)
def load_testing_data() -> pd.DataFrame:
    return pd.read_csv(TEST_DATA_PATH)


def build_input_row() -> pd.DataFrame:
    now_epoch = int(pd.Timestamp.now().timestamp())

    with st.form("traffic_form"):
        left, middle, right = st.columns(3)

        with left:
            ts = st.number_input("Timestamp", value=now_epoch, step=1)
            id_orig_h = st.text_input("Origin IP", value="192.168.1.10")
            id_orig_p = st.number_input("Origin Port", min_value=0, max_value=65535, value=51524)
            proto = st.selectbox("Protocol", ["tcp", "udp", "icmp"])
            duration = st.number_input("Duration", min_value=0.0, value=1.25, step=0.01)
            orig_pkts = st.number_input("Origin Packets", min_value=0, value=8, step=1)

        with middle:
            id_resp_h = st.text_input("Response IP", value="192.168.1.1")
            id_resp_p = st.number_input("Response Port", min_value=0, max_value=65535, value=80)
            service = st.selectbox("Service", ["http", "dns", "ssl", "ssh", "dhcp", "irc", "unknown"])
            conn_state = st.selectbox("Connection State", ["SF", "S0", "REJ", "RSTO", "RSTR", "OTH", "SH", "SHR"])
            resp_pkts = st.number_input("Response Packets", min_value=0, value=6, step=1)
            missed_bytes = st.number_input("Missed Bytes", min_value=0, value=0, step=1)

        with right:
            orig_bytes = st.number_input("Origin Bytes", min_value=0.0, value=350.0, step=1.0)
            resp_bytes = st.number_input("Response Bytes", min_value=0.0, value=1200.0, step=1.0)
            orig_ip_bytes = st.number_input("Origin IP Bytes", min_value=0, value=650, step=1)
            resp_ip_bytes = st.number_input("Response IP Bytes", min_value=0, value=1500, step=1)
            history = st.text_input("History", value="ShADadFf")
            local_flags = st.columns(2)
            local_orig = local_flags[0].selectbox("Local Orig", ["-", "T", "F"])
            local_resp = local_flags[1].selectbox("Local Resp", ["-", "T", "F"])

        submitted = st.form_submit_button("Predict Traffic", type="primary", use_container_width=True)

    row = pd.DataFrame(
        [
            {
                "ts": ts,
                "uid": "manual-input",
                "id.orig_h": id_orig_h,
                "id.orig_p": id_orig_p,
                "id.resp_h": id_resp_h,
                "id.resp_p": id_resp_p,
                "proto": proto,
                "service": np.nan if service == "unknown" else service,
                "duration": duration,
                "orig_bytes": orig_bytes,
                "resp_bytes": resp_bytes,
                "conn_state": conn_state,
                "local_orig": np.nan if local_orig == "-" else int(local_orig == "T"),
                "local_resp": np.nan if local_resp == "-" else int(local_resp == "T"),
                "missed_bytes": missed_bytes,
                "history": np.nan if not history.strip() else history.strip(),
                "orig_pkts": orig_pkts,
                "orig_ip_bytes": orig_ip_bytes,
                "resp_pkts": resp_pkts,
                "resp_ip_bytes": resp_ip_bytes,
                "label": "manual",
            }
        ],
        columns=RAW_COLUMNS,
    )

    for col in NUMERIC_COLUMNS:
        row[col] = pd.to_numeric(row[col], errors="coerce")

    row.attrs["submitted"] = submitted
    return row


def build_testing_data_row() -> pd.DataFrame:
    if not TEST_DATA_PATH.exists():
        st.error(f"Testing data file not found: {TEST_DATA_PATH}")
        empty = pd.DataFrame()
        empty.attrs["submitted"] = False
        return empty

    try:
        test_df = load_testing_data()
    except Exception as error:
        st.error("Could not load testing_data.csv.")
        st.exception(error)
        empty = pd.DataFrame()
        empty.attrs["submitted"] = False
        return empty

    row_count = len(test_df)
    if row_count == 0:
        st.error("testing_data.csv is empty.")
        empty = pd.DataFrame()
        empty.attrs["submitted"] = False
        return empty

    csv_left, csv_right = st.columns([1, 2])
    with csv_left:
        selected_index = st.number_input(
            "Testing Data Row",
            min_value=0,
            max_value=row_count - 1,
            value=0,
            step=1,
        )
        submitted = st.button("Predict Selected Row", type="primary", use_container_width=True)

    selected_row = test_df.iloc[[int(selected_index)]].reset_index(drop=True)
    with csv_right:
        st.dataframe(selected_row, use_container_width=True)

    if "traffic_type" in selected_row.columns:
        st.info(f"Actual Label: {selected_row.loc[0, 'traffic_type']}")

    selected_row.attrs["submitted"] = submitted
    return selected_row


def safe_number(value: Any) -> float:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return 0.0
    return float(number)


def show_prediction(label: str, status: str) -> None:
    if status == "malicious":
        st.error(f"Prediction: {label}")
    elif status == "benign":
        st.success(f"Prediction: {label}")
    else:
        st.warning(f"Prediction: {label}")


def show_visuals(raw_input: pd.DataFrame, engineered_input: pd.DataFrame, scores: pd.DataFrame | None) -> None:
    visual_left, visual_right = st.columns(2)

    bytes_df = pd.DataFrame(
        {
            "direction": ["Origin Bytes", "Response Bytes"],
            "bytes": [
                safe_number(raw_input.loc[0, "orig_bytes"]),
                safe_number(raw_input.loc[0, "resp_bytes"]),
            ],
        }
    )
    packet_df = pd.DataFrame(
        {
            "direction": ["Origin Packets", "Response Packets"],
            "packets": [
                int(safe_number(raw_input.loc[0, "orig_pkts"])),
                int(safe_number(raw_input.loc[0, "resp_pkts"])),
            ],
        }
    )

    with visual_left:
        st.plotly_chart(
            px.bar(bytes_df, x="direction", y="bytes", color="direction", title="Byte Volume"),
            use_container_width=True,
        )
        st.plotly_chart(
            px.bar(packet_df, x="direction", y="packets", color="direction", title="Packet Volume"),
            use_container_width=True,
        )

    with visual_right:
        if scores is not None and not scores.empty:
            st.plotly_chart(
                px.bar(scores, x="label", y="confidence", color="label", range_y=[0, 1], title="Prediction Confidence"),
                use_container_width=True,
            )

        port_df = pd.DataFrame(
            {
                "port_type": ["Origin Port", "Response Port"],
                "port": [int(safe_number(raw_input.loc[0, "id.orig_p"])), int(safe_number(raw_input.loc[0, "id.resp_p"]))],
            }
        )
        st.plotly_chart(
            px.bar(port_df, x="port_type", y="port", color="port_type", title="Port Summary"),
            use_container_width=True,
        )

    st.subheader("Input Record")
    st.dataframe(raw_input.drop(columns=["uid", "label"], errors="ignore"), use_container_width=True)

    st.subheader("Engineered Features")
    st.dataframe(engineered_input, use_container_width=True)


model, model_error = load_model()

with st.sidebar:
    st.title("Model Status")
    if model_error:
        st.error("Model not ready")
    else:
        st.success("Model loaded")
        st.caption(type(model).__name__)

    st.divider()
    st.title("Project")
    st.write("MLC Network Detection System")
    st.write("Manual IoT-23 traffic prediction")

    st.divider()
    st.title("Labels")
    st.write("Benign: normal traffic")
    st.write("Malicious: anomaly or attack traffic")


st.title("MLC Network Detection System")
st.caption("Manual network traffic testing with the saved anomaly detection model.")

if model_error:
    st.error(model_error)
    st.stop()

input_source = st.radio(
    "Prediction Source",
    ["Manual Form", "testing_data.csv Row"],
    horizontal=True,
)

if input_source == "Manual Form":
    raw_input = build_input_row()
else:
    raw_input = build_testing_data_row()

if raw_input.attrs.get("submitted") and not raw_input.empty:
    try:
        engineered_input = prepare_model_features(raw_input)
        prediction = model.predict(engineered_input)[0]
        label, status = normalize_prediction(prediction)
        scores = get_prediction_scores(model, engineered_input)

        show_prediction(label, status)
        if "traffic_type" in raw_input.columns:
            actual_label = raw_input.loc[0, "traffic_type"]
            if str(actual_label).strip():
                st.write(f"Actual Label: `{actual_label}`")
        show_visuals(raw_input, engineered_input, scores)
    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)
        st.info(
            "If this error mentions missing columns or preprocessing, save a complete sklearn "
            "Pipeline from the notebook so the app receives the same feature layout used during training."
        )

import gradio as gr
import pandas as pd
import pickle

# Load model and features
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Prediction logic
def predict_price(city, province, property_type, purpose, location_id, latitude, longitude, baths, bedrooms, area_size):
    warnings = []

    if area_size > 25:
        warnings.append("<span style='color:red;'>⚠️ Area size exceeds typical range — prediction may be inaccurate.</span>")
    if bedrooms > 7:
        warnings.append("<span style='color:red;'>⚠️ Bedrooms count unusually high — results might not reflect real value.</span>")
    if baths > 7:
        warnings.append("<span style='color:red;'>⚠️ Number of bathrooms is very high — prediction may be off.</span>")

    input_dict = {
        'city': city,
        'province_name': province,
        'property_type': property_type,
        'purpose': purpose,
        'location_id': int(location_id),
        'latitude': float(latitude),
        'longitude': float(longitude),
        'baths': int(baths),
        'bedrooms': int(bedrooms),
        'Area Size': float(area_size)
    }

    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    prediction = model.predict(df)[0]
    
    celebration = "<div style='font-size: 50px; text-align: center;'>🎉</div>"
    result = celebration + f"<h3 style='color:#00e676;'>🏠 Predicted Price: PKR {int(prediction):,}</h3>"

    if warnings:
        result += "<ul>" + "".join(f"<li>{w}</li>" for w in warnings) + "</ul>"

    return result

# Custom CSS for styling
custom_css = """
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
    color: white !important;
}
h1 {
    font-size: 2.5em;
    color: #00e676;
    text-align: center;
    animation: glow 2s infinite;
}
@keyframes glow {
    0% { text-shadow: 0 0 5px #00e676; }
    50% { text-shadow: 0 0 20px #00e676; }
    100% { text-shadow: 0 0 5px #00e676; }
}
.gr-button {
    background-color: #00e676 !important;
    color: black !important;
    font-weight: bold;
}
.gr-box, .gr-input, .gr-textbox, .gr-dropdown, .gr-slider {
    background-color: #1c1c1c !important;
    color: white !important;
}
"""

# UI layout
with gr.Blocks(css=custom_css) as ui:
    gr.HTML("<h1>🏠 House Price Predictor</h1>")

    with gr.Row():
        with gr.Column():
            city = gr.Dropdown(['Islamabad', 'Lahore', 'Karachi'], label="City")
            province = gr.Dropdown(['Islamabad Capital', 'Punjab', 'Sindh'], label="Province")
            property_type = gr.Dropdown(['House', 'Flat', 'Upper Portion'], label="Property Type")
            purpose = gr.Dropdown(['For Sale', 'For Rent'], label="Purpose")
            location_id = gr.Number(label="Location ID")
        with gr.Column():
            latitude = gr.Slider(30.0, 35.0, step=0.0001, label="Latitude")
            longitude = gr.Slider(60.0, 75.0, step=0.0001, label="Longitude")
            baths = gr.Slider(1, 10, step=1, label="Baths")
            bedrooms = gr.Slider(1, 10, step=1, label="Bedrooms")
            area_size = gr.Slider(1, 40, step=0.1, label="Area Size (Marla/Kanal)")

    predict_btn = gr.Button("💡 Predict Price")
    output = gr.HTML(label="Estimated Price")

    predict_btn.click(
        fn=predict_price,
        inputs=[city, province, property_type, purpose, location_id, latitude, longitude, baths, bedrooms, area_size],
        outputs=output
    )

ui.launch(share=True)

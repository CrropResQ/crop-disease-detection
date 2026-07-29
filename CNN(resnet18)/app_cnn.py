import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models

# -------- LOAD MODEL --------
@st.cache_resource
def load_model(num_classes):

    model = models.resnet18(pretrained=False)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        num_classes
    )

    model.load_state_dict(
        torch.load("models/cnn.pth", map_location="cpu")
    )

    model.eval()

    return model

# -------- CLASS LABELS --------
classes = [
    "Late Blight (Tomato)",
    "Healthy (Tomato)"
]

model = load_model(len(classes))

# -------- IMAGE TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------- UI --------
st.title("🌿 Crop Disease Detection (CNN Model)")

uploaded_file = st.file_uploader("Upload leaf image")

temp = st.slider("Temperature", 10, 45)
humidity = st.slider("Humidity", 10, 100)

# -------- PREDICTION --------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)

        probs = torch.softmax(outputs, dim=1)

        conf, pred = torch.max(probs, 1)

    disease = classes[pred.item()]

    st.write(f"### Disease: {disease}")

    st.write(f"Confidence: {conf.item()*100:.2f}%")

    if humidity > 70:
        st.warning("⚠ High humidity → fungal disease risk")

    st.write("### Advisory")

    if "Late" in disease:

        st.write("- Apply fungicide")
        st.write("- Remove infected leaves")

    else:

        st.write("- Plant is healthy")
        st.write("- Maintain proper irrigation")

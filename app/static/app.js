const form = document.getElementById("upload-form");
const imageInput = document.getElementById("image-input");
const resultSection = document.getElementById("result");
const errorSection = document.getElementById("error");
const uploadedImage = document.getElementById("uploaded-image");
const resultImage = document.getElementById("result-image");
const resultFile = document.getElementById("result-file");
const resultDistance = document.getElementById("result-distance");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorSection.classList.add("hidden");
  resultSection.classList.add("hidden");

  if (!imageInput.files.length) {
    errorSection.textContent = "Choose an image first.";
    errorSection.classList.remove("hidden");
    return;
  }

  const formData = new FormData();
  formData.append("image", imageInput.files[0]);

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Search request failed.");
    }

    const data = await response.json();
    uploadedImage.src = data.uploaded_image;
    resultImage.src = data.match;
    resultFile.textContent = `Match: ${data.match}`;
    resultDistance.textContent = `Distance: ${data.distance.toFixed(4)}`;
    resultSection.classList.remove("hidden");
  } catch (err) {
    errorSection.textContent = err.message;
    errorSection.classList.remove("hidden");
  }
});

// Set active nav link
const currentPath = window.location.pathname;
document.querySelectorAll(".nav-link").forEach((link) => {
  if (link.getAttribute("href") === currentPath) {
    link.classList.add("active");
  } else {
    link.classList.remove("active");
  }
});

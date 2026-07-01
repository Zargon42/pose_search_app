const form = document.getElementById("upload-form");
const imageInput = document.getElementById("image-input");
const clearBtn = document.getElementById("clear-btn");
const resultSection = document.getElementById("result");
const errorSection = document.getElementById("error");
const resultMessage = document.getElementById("result-message");

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
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Upload request failed.");
    }

    const data = await response.json();
    resultMessage.textContent = data.message;
    resultSection.classList.remove("hidden");
    imageInput.value = ""; // Clear the file input
  } catch (err) {
    errorSection.textContent = err.message;
    errorSection.classList.remove("hidden");
  }
});

clearBtn.addEventListener("click", async (event) => {
  event.preventDefault();
  
  if (!confirm("Are you sure you want to delete all uploaded images and the index? This cannot be undone.")) {
    return;
  }

  errorSection.classList.add("hidden");
  resultSection.classList.add("hidden");

  try {
    const response = await fetch("/api/clear-index", {
      method: "POST",
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Clear request failed.");
    }

    const data = await response.json();
    resultMessage.textContent = data.message;
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

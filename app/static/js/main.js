// main.js

/**
 * Disables the checkout button and shows a spinner to indicate processing.
 * Prevents multiple form submissions.
 * @param {HTMLFormElement} form - The checkout form element.
 */
function handleCheckoutSubmit(form) {
  const btn = document.getElementById("checkout-btn");

  if (form && btn && !btn.disabled) {
    // Disable the button and show spinner
    btn.disabled = true;
    btn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Processing...
        `;
    // Prevent double submit fallback
    setTimeout(() => {
      btn.disabled = true;
    }, 500);
  }
  // Optional: Scroll to top for feedback
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/**
 * Auto-dismiss Bootstrap alerts after 4 seconds.
 */
document.addEventListener("DOMContentLoaded", function () {
  setTimeout(function () {
    document.querySelectorAll(".alert").forEach(function (alert) {
      alert.classList.add("fade");
      setTimeout(() => alert.remove(), 500);
    });
  }, 4000);
});

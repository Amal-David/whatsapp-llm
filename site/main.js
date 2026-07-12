const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
document.documentElement.classList.add("motion-ready");

const header = document.querySelector("[data-header]");
const updateHeader = () => header?.classList.toggle("is-scrolled", window.scrollY > 24);
updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

const revealItems = document.querySelectorAll(".reveal");
if (reducedMotion || !("IntersectionObserver" in window)) {
  revealItems.forEach((item) => item.classList.add("is-visible"));
} else {
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.14 }
  );
  revealItems.forEach((item, index) => {
    item.classList.add(`reveal-delay-${Math.min(index % 4, 3)}`);
    revealObserver.observe(item);
  });
}

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.querySelector(button.dataset.copy);
    if (!target) return;
    const original = button.textContent;
    try {
      await navigator.clipboard.writeText(target.textContent.trim());
      button.textContent = "Copied";
    } catch {
      button.textContent = "Select text";
    }
    window.setTimeout(() => {
      button.textContent = original;
    }, 1600);
  });
});

const docLinks = [...document.querySelectorAll("[data-doc-link]")];
const docSections = docLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);
if (docLinks.length && docSections.length && "IntersectionObserver" in window) {
  const docObserver = new IntersectionObserver(
    (entries) => {
      const visible = entries.find((entry) => entry.isIntersecting);
      if (!visible) return;
      docLinks.forEach((link) => {
        const active = link.getAttribute("href") === `#${visible.target.id}`;
        link.classList.toggle("is-active", active);
        if (active) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
      });
    },
    { rootMargin: "-16% 0px -70%", threshold: 0 }
  );
  docSections.forEach((section) => docObserver.observe(section));
}

const canvas = document.querySelector("[data-memory-field]");
if (canvas) {
  const context = canvas.getContext("2d");
  const points = Array.from({ length: 26 }, (_, index) => ({
    x: (index * 47) % 100 / 100,
    y: (index * 71) % 100 / 100,
    drift: 0.00004 + (index % 5) * 0.000012,
    phase: index * 0.67,
  }));
  let width = 0;
  let height = 0;
  let frame = 0;

  const resizeCanvas = () => {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    width = rect.width;
    height = rect.height;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
  };

  const drawField = (time = 0) => {
    context.clearRect(0, 0, width, height);
    const positions = points.map((point) => ({
      x: point.x * width + Math.sin(time * point.drift + point.phase) * 13,
      y: point.y * height + Math.cos(time * point.drift * 0.8 + point.phase) * 9,
    }));
    positions.forEach((point, index) => {
      positions.slice(index + 1).forEach((other) => {
        const distance = Math.hypot(point.x - other.x, point.y - other.y);
        if (distance < 165) {
          context.strokeStyle = `rgba(121, 241, 90, ${0.12 * (1 - distance / 165)})`;
          context.lineWidth = 0.7;
          context.beginPath();
          context.moveTo(point.x, point.y);
          context.lineTo(other.x, other.y);
          context.stroke();
        }
      });
      context.fillStyle = index % 9 === 0 ? "rgba(255, 107, 87, 0.8)" : "rgba(242, 239, 231, 0.55)";
      context.fillRect(point.x - 1, point.y - 1, 2, 2);
    });
    if (!reducedMotion) frame = window.requestAnimationFrame(drawField);
  };

  resizeCanvas();
  drawField();
  window.addEventListener("resize", resizeCanvas, { passive: true });
  window.addEventListener("pagehide", () => window.cancelAnimationFrame(frame), { once: true });
}

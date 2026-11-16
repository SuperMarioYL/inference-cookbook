for (const figure of document.querySelectorAll("[data-mechanism]")) {
  const frames = [...figure.querySelectorAll("svg [data-frame]")];
  const events = [...figure.querySelectorAll("svg [data-event]")];
  const captions = [...figure.querySelectorAll("figcaption li")];
  const controls = figure.querySelector(".mechanism-controls");
  const counter = controls.querySelector(".step-counter");
  const back = controls.querySelector("[data-back]");
  const next = controls.querySelector("[data-next]");
  const play = controls.querySelector("[data-play]");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  let step = 0;
  let timer;

  function pause() {
    clearInterval(timer);
    timer = undefined;
    play.textContent = step === captions.length - 1 ? "重播" : "播放";
  }

  function show(index, moving = true) {
    const previous = step;
    step = index;
    captions.forEach((caption, i) => { caption.hidden = i !== step; });
    frames.forEach((frame, i) => { frame.style.display = i === step ? "inline" : "none"; });
    for (const [i, event] of events.entries()) {
      event.classList.toggle("current-event", i === step);
      event.classList.toggle("future-event", i > step);
    }
    counter.textContent = `${step + 1} / ${captions.length}`;
    back.disabled = step === 0;
    next.disabled = step === captions.length - 1;
    if (next.disabled) pause();
    else if (!timer) play.textContent = "播放";

    if (reducedMotion.matches || !moving) return;
    if (events.length) {
      events[step].querySelector(".message-edge").animate([
        { strokeDasharray: "1 1", strokeDashoffset: 1 },
        { strokeDasharray: "1 1", strokeDashoffset: 0 },
      ], { duration: 500, easing: "ease-out" });
    } else {
      const cells = frames[step].querySelectorAll(".state-cell");
      if (cells.length) {
        const oldCells = frames[previous].querySelectorAll(".state-cell");
        cells.forEach((cell, i) => {
          if (cell.textContent.trim() !== oldCells[i].textContent.trim()) {
            cell.querySelector("text").animate([
              { transform: "translateY(3px)", opacity: 0.35 },
              { transform: "translateY(0)", opacity: 1 },
            ], { duration: 240, easing: "ease-out" });
          }
        });
      } else {
        frames[step].animate([{ opacity: 0.45 }, { opacity: 1 }], { duration: 180 });
      }
      const cursor = frames[step].querySelector(".time-cursor");
      if (cursor) {
        const oldX = Number(frames[previous].querySelector(".time-cursor").dataset.x);
        cursor.animate([
          { transform: `translateX(${oldX - Number(cursor.dataset.x)}px)` },
          { transform: "translateX(0)" },
        ], { duration: 450, easing: "ease-out" });
      }
      for (const packet of frames[step].querySelectorAll(".packet")) {
        const { dx, dy } = packet.dataset;
        packet.animate([
          { transform: `translate(${dx}px, ${dy}px)` },
          { transform: "translate(0, 0)" },
        ], { duration: 800, easing: "ease-in-out" });
      }
    }
  }

  back.onclick = () => { pause(); show(step - 1); };
  next.onclick = () => { pause(); show(step + 1); };
  play.onclick = () => {
    if (timer) { pause(); return; }
    if (step === captions.length - 1) show(0);
    timer = setInterval(() => show(step + 1), 3200);
    play.textContent = "暂停";
  };
  document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });
  reducedMotion.addEventListener("change", pause);
  show(0, false);
  controls.hidden = false;
}

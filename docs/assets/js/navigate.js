document.addEventListener("keydown", (event) => {
  const match = window.location.pathname.match(/day(\d+)\.html/);

  if (match) {
    let currentPage = parseInt(match[1], 10);

    const minPage = 0;
    const maxPage = 99;

    if (event.ctrlKey) {
      if (event.key === "ArrowRight" && currentPage < maxPage) {
        window.location.href = `day${currentPage + 1}.html`;
      } else if (event.key === "ArrowLeft" && currentPage > minPage) {
        window.location.href = `day${currentPage - 1}.html`;
      }
    }
  }
});

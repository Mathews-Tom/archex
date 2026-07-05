function paginateCatalog(items, pageSize) {
  const pages = [];
  for (let i = 0; i < items.length; i += pageSize) {
    pages.push(items.slice(i, i + pageSize));
  }
  return pages;
}

function sortBySeason(items, season) {
  return items.filter((item) => item.season === season);
}

function filterByPriceTier(items, tier) {
  return items.filter((item) => item.priceTier === tier);
}

function reorderMerchandisingCategories(categories, promotedCategoryId) {
  const promoted = categories.find((category) => category.id === promotedCategoryId);
  const rest = categories.filter((category) => category.id !== promotedCategoryId);
  return promoted ? [promoted, ...rest] : categories;
}

function fetchProductSizingChart(productId) {
  return { productId, sizes: ["XS", "S", "M", "L", "XL"] };
}

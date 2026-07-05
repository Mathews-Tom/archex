function calculateOrderTotal(lineItems, shippingFee) {
  const subtotal = lineItems.reduce((sum, item) => sum + item.price * item.qty, 0);
  return subtotal + shippingFee;
}

function validateShippingAddress(address) {
  return Boolean(address.street && address.city && address.postalCode);
}

function isServiceableCarrierZone(postalCode, carrierZones) {
  return carrierZones.some((zone) => zone.postalCodes.includes(postalCode));
}

function generateOrderConfirmationNumber(cartId) {
  return `ORD-${cartId}-${Date.now().toString(36).toUpperCase()}`;
}

function estimateDeliveryWindow(shippingSpeed) {
  const speedToDays = { standard: 5, expedited: 2, overnight: 1 };
  return speedToDays[shippingSpeed] ?? speedToDays.standard;
}
